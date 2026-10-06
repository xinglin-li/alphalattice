"""A model's search axes: the ordered space a search may propose from (EX, its design's section 5).

The format V309's proposers read. Each axis names one recipe parameter, its kind, its bounds or
choices, whether a numeric axis is searched on a log scale, an optional step, the default (the
reference recipe's value) and, for a conditional axis, the earlier categorical axis and the
choices it exists under. This wave the axes are declared, validated and held by the contract
suite; they enter no identity, and the search domain a mandate seals stays its own.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type AxisKind = Literal["int", "float", "categorical", "bool"]
type AxisValue = bool | int | float | str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class AxisCondition(_Contract):
    """The earlier categorical axis, and the choices of it this axis exists under."""

    axis: str = Field(min_length=1, description="The earlier categorical axis.")
    choices: tuple[AxisValue, ...] = Field(
        alias="in", min_length=1, description="The parent's choices this axis exists under."
    )


class SearchAxis(_Contract):
    """One searchable recipe parameter."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="The recipe parameter.")
    kind: AxisKind = Field(description="`int`, `float`, `categorical` or `bool`.")
    low: float | None = Field(default=None, description="A numeric axis's lower bound.")
    high: float | None = Field(default=None, description="A numeric axis's upper bound.")
    log: bool = Field(default=False, description="Whether a numeric axis is searched in log.")
    choices: tuple[AxisValue, ...] | None = Field(
        default=None, description="A categorical axis's choices, or a numeric axis's set."
    )
    step: float | None = Field(default=None, gt=0, description="A numeric axis's grid step.")
    default: AxisValue = Field(description="The reference recipe's value.")
    when: AxisCondition | None = Field(
        default=None, description="The earlier categorical axis this axis exists under."
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _shape(self) -> Self:
        numeric = self.kind in {"int", "float"}
        bounded = self.low is not None or self.high is not None
        if not numeric and (bounded or self.log or self.step is not None):
            raise ValueError(f"search_axes.numeric_field_on_{self.kind}:{self.name}")
        if self.kind == "categorical" and not self.choices:
            raise ValueError(f"search_axes.choices_required:{self.name}")
        if self.kind == "bool" and self.choices is not None:
            raise ValueError(f"search_axes.bool_takes_no_choices:{self.name}")
        if numeric and self.choices is None:
            if self.low is None or self.high is None or not self.low < self.high:
                raise ValueError(f"search_axes.bounds_invalid:{self.name}")
            if self.log and self.low <= 0:
                raise ValueError(f"search_axes.log_bounds_invalid:{self.name}")
        if numeric and self.choices is not None and (bounded or self.log):
            raise ValueError(f"search_axes.set_takes_no_bounds:{self.name}")
        values = (*(self.choices or ()), *(v for v in (self.low, self.high) if v is not None))
        if numeric and not all(_number(v) for v in values):
            raise ValueError(f"search_axes.value_not_numeric:{self.name}")
        if self.kind == "int" and not all(float(v).is_integer() for v in values):
            raise ValueError(f"search_axes.value_not_integral:{self.name}")
        if self.choices is not None and len(set(self.choices)) != len(self.choices):
            raise ValueError(f"search_axes.choices_duplicated:{self.name}")
        if not self.admits(self.default):
            raise ValueError(f"search_axes.default_outside_axis:{self.name}")
        return self

    def admits(self, value: object) -> bool:
        """Whether the axis holds the value: in its choices, or in its bounds and on its step.

        Args:
            value: A parameter value.

        Returns:
            Whether the value lies on the axis.
        """
        if self.kind == "bool":
            return isinstance(value, bool)
        if self.choices is not None:
            return any(type(value) is type(v) and value == v for v in self.choices)
        if not _number(value) or (self.kind == "int" and not isinstance(value, int)):
            return False
        assert self.low is not None and self.high is not None
        number = float(value)  # type: ignore[arg-type]
        if not self.low <= number <= self.high:
            return False
        if self.step is None:
            return True
        offset = (number - self.low) / self.step
        return math.isclose(offset, round(offset), abs_tol=1e-9)

    def extremes(self) -> tuple[AxisValue, ...]:
        """The values a contract probes on this axis: its choices, or its bounds.

        Returns:
            The choices in order, both booleans, or the low and high bounds.
        """
        if self.kind == "bool":
            return (False, True)
        if self.choices is not None:
            return self.choices
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            return (int(self.low), int(self.high))
        return (self.low, self.high)

    def draw(self, generator: random.Random) -> AxisValue:
        """One value on the axis, uniform on its scale and snapped to its step.

        Args:
            generator: The seeded generator a sample draws from.

        Returns:
            The value.
        """
        if self.kind == "bool" or self.choices is not None:
            return generator.choice(self.extremes())
        assert self.low is not None and self.high is not None
        if self.log:
            value = math.exp(generator.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = generator.uniform(self.low, self.high)
        if self.step is not None:
            value = self.low + round((value - self.low) / self.step) * self.step
        value = min(max(value, self.low), self.high)
        return round(value) if self.kind == "int" else value


class SearchSpace(_Contract):
    """A model's ordered axes: the order is the domain's, and a `when` names an earlier axis."""

    axes: tuple[SearchAxis, ...] = Field(min_length=1, description="The axes, in order.")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _order(self) -> Self:
        seen: dict[str, SearchAxis] = {}
        for axis in self.axes:
            if axis.name in seen:
                raise ValueError(f"search_axes.axis_duplicated:{axis.name}")
            if axis.when is not None:
                parent = seen.get(axis.when.axis)
                # One level: the parent is an earlier, unconditional categorical axis.
                if parent is None or parent.kind != "categorical" or parent.when is not None:
                    raise ValueError(f"search_axes.condition_invalid:{axis.name}")
                if not all(parent.admits(value) for value in axis.when.choices):
                    raise ValueError(f"search_axes.condition_choice_unknown:{axis.name}")
            seen[axis.name] = axis
        return self

    def holds(self, axis: SearchAxis, point: Mapping[str, AxisValue]) -> bool:
        """Whether an axis exists at a point.

        It is unconditional, or its parent's choice is one its `when` names.

        Args:
            axis: One of the space's axes.
            point: The parent axes' values.

        Returns:
            Whether the axis is active.
        """
        return axis.when is None or point.get(axis.when.axis) in axis.when.choices

    def complete(self, point: Mapping[str, AxisValue]) -> dict[str, AxisValue]:
        """A point with every active axis the point leaves out at its default, and no inactive one.

        Args:
            point: Some axes' values.

        Returns:
            The active axes' values, in the space's order.
        """
        values: dict[str, AxisValue] = {}
        for axis in self.axes:
            if self.holds(axis, values):
                values[axis.name] = point.get(axis.name, axis.default)
        return values

    def probes(self, *, samples: int = 16, seed: int = 0) -> tuple[dict[str, AxisValue], ...]:
        """The points a contract probes.

        The defaults; each axis at each of its extremes with the rest at their defaults (a
        conditional axis under its parent's first admitting choice); every axis at its first
        and at its last extreme; and a seeded sample.

        Args:
            samples: How many seeded points.
            seed: The sample's seed.

        Returns:
            The points, each complete and holding only its active axes.
        """
        points = [self.complete({})]
        for axis in self.axes:
            parent = {} if axis.when is None else {axis.when.axis: axis.when.choices[0]}
            points.extend(self.complete({**parent, axis.name: v}) for v in axis.extremes())
        for side in (0, -1):
            points.append(self.complete({v.name: v.extremes()[side] for v in self.axes}))
        generator = random.Random(seed)
        for _ in range(samples):
            drawn: dict[str, AxisValue] = {}
            for axis in self.axes:
                if self.holds(axis, drawn):
                    drawn[axis.name] = axis.draw(generator)
            points.append(drawn)
        unique: dict[tuple[tuple[str, AxisValue], ...], dict[str, AxisValue]] = {}
        for point in points:
            unique.setdefault(tuple(point.items()), point)
        return tuple(unique.values())


__all__ = ["AxisCondition", "AxisKind", "AxisValue", "SearchAxis", "SearchSpace"]
