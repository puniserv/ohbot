# Run from the repo root: uv run --group build pyinstaller packaging/ohbotchat.spec
from PyInstaller.utils.hooks import collect_data_files

ROOT = SPECPATH + "/.."

a = Analysis(
    [f"{ROOT}/packaging/entry.py"],
    pathex=[ROOT],
    datas=[
        (f"{ROOT}/web", "web"),
        (f"{ROOT}/personas", "personas"),
        (f"{ROOT}/config.toml", "."),
        (f"{ROOT}/ohbotchat/models", "ohbotchat/models"),
        *collect_data_files("ohbot"),
    ],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ohbot-chat",
    console=True,
    # everything lands next to the exe, which is where config.app_root() looks
    contents_directory=".",
)
coll = COLLECT(exe, a.binaries, a.datas, name="ohbot-chat")
