// The UI audit and isolated Node readers use the Python catalog owner's exact projection.
const path = require('node:path');
let words;
module.exports = function messageWords() {
  if (!words) {
    const root = path.resolve(__dirname, '../../../..');
    const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const read = require('node:child_process').spawnSync(python,
      ['-B', '-c', 'import json; from alphalattice.interface.local_application.message_catalog import catalog_words; print(json.dumps(catalog_words()))'],
      {cwd:root, encoding:'utf8', windowsHide:true,
       env:{...process.env, PYTHONPATH:path.join(root,'src'), ALPHALATTICE_NETWORK_DISABLED:'1'}});
    if (read.status !== 0) throw Error('message_catalog.projection_unreadable: ' + (read.stderr || read.error));
    words = Object.freeze(JSON.parse(read.stdout));
  }
  return words;
};
