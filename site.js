"use strict";

const copyButton = document.getElementById("copy-intent");
const copyStatus = document.getElementById("copy-status");
let copyReset;
copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  clearTimeout(copyReset);
  try {
    await navigator.clipboard.writeText(document.getElementById("intent-text").textContent.trim());
    copyButton.textContent = "✓ Copied!";
    copyButton.dataset.copied = "true";
    copyStatus.textContent = "";
    copyButton.setAttribute("aria-label", "Message copied to clipboard");
    copyReset = setTimeout(() => {
      copyButton.textContent = "Copy message";
      copyButton.removeAttribute("data-copied");
      copyButton.removeAttribute("aria-label");
    }, 2500);
  } catch {
    copyButton.textContent = "Copy message";
    copyButton.removeAttribute("data-copied");
    copyButton.removeAttribute("aria-label");
    copyStatus.textContent = "Select and copy the message above.";
  }
});

const viewRecord = document.querySelector(".workbench-record");
const tabList = document.querySelector(".view-tabs");
const tabs = [...tabList.querySelectorAll("a")];
const panels = tabs.map(tab => document.querySelector(tab.getAttribute("href")));
function selectView(tab, focus = false) {
  tabs.forEach((item, index) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    panels[index].hidden = !selected;
  });
  if (focus) tab.focus();
}
tabList.setAttribute("role", "tablist");
viewRecord.dataset.enhanced = "true";
tabs.forEach((tab, index) => {
  tab.setAttribute("role", "tab");
  tab.setAttribute("aria-controls", panels[index].id);
  panels[index].setAttribute("role", "tabpanel");
  panels[index].tabIndex = 0;
  tab.addEventListener("click", event => {
    if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    selectView(tab);
  });
  tab.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else if (event.key === " ") next = index;
    else return;
    event.preventDefault();
    selectView(tabs[next], true);
  });
});
selectView(tabs.find(tab => tab.getAttribute("href") === location.hash) || tabs[0]);

const viewer = document.getElementById("image-viewer");
const zoomImage = document.getElementById("zoom-image");
let zoomOrigin;
for (const link of document.querySelectorAll("[data-zoom]")) {
  link.addEventListener("click", event => {
    if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || typeof viewer.showModal !== "function") return;
    event.preventDefault();
    zoomOrigin = link;
    zoomImage.src = link.getAttribute("href");
    zoomImage.alt = link.querySelector("img")?.alt
      || link.closest(".view-panel")?.querySelector("img")?.alt
      || "Full-size Workbench screenshot.";
    zoomImage.removeAttribute("width");
    zoomImage.removeAttribute("height");
    viewer.showModal();
  });
}
document.getElementById("close-image").addEventListener("click", () => viewer.close());
viewer.addEventListener("close", () => zoomOrigin?.focus());
