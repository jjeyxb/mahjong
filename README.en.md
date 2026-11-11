# MIA — Mahjong Soul Assistant

[繁體中文](README.md)

MIA (Mahjong Intelligence Assistant) is a desktop app that gives you advice while you play Mahjong Soul. It reads the game screen or the game data and tells you what to discard, how far your hand is from tenpai, and which tiles cannot deal in. It runs on Windows and Apple Silicon Macs, and works with both the web and Steam versions of Mahjong Soul.

MIA is a university capstone project. The interface is available in Traditional Chinese and English.

> [!WARNING]
> **Third-party assistance tools may violate Mahjong Soul's terms of service, and your account may be penalized. You use MIA at your own risk.**
> This project is for academic research. Use it for spectating, replays or games against bots, **not in ranked games**. See [Things to know](#things-to-know).

## Features

| Feature | What it gives you |
|---|---|
| **AI advice** | Advice from the [Mortal](https://github.com/Equim-chan/Mortal) mahjong AI for the current turn: what to discard, whether to call, riichi or win, with a score for each option. |
| **Shanten analysis** | Reads your hand and shows the shanten number and the tile acceptance after each discard. |
| **Deal-in analysis** | Only shows what follows from the rules: how many waits remain on a tile after genbutsu, suji, kabe and furiten are ruled out. No guessed percentages. |

- **Five play styles**: Standard, plus four styles fine-tuned in this project: Calls more, Calls less, Riichi more, Riichi less. You can run a second style side by side for comparison.
- **Web and Steam**: for the web version, MIA opens the game in a bundled browser; for Steam, it reads the game window.
- **Three recognition methods**: Template matching, YOLO and Hybrid, chosen on each feature's settings page.
- **Overlay**: advice can be drawn directly over the table, so you can see it in a full-screen game.
- **Interface language**: Traditional Chinese or English.

Each feature has its own switch and works independently, so turn on only what you need.

## System requirements

- **Windows**: 64-bit (x64).
- **Mac**: Apple Silicon (M-series chip). Intel Macs are not supported.
- **Disk space**: about 3 GB after installation (models and the browser are included).
- The installer includes everything MIA needs. **You do not need to install Python or any other tools.**

## Download and install

Download the installer for your system from [Releases](https://github.com/jjeyxb/mahjong/releases). These are preview builds and are not signed by Microsoft or Apple.

### Windows

1. Download `MIA-<version>-windows-x64-setup.exe`. If the release also has `.bin` files, **download all of them into the same folder**; the installer needs them.
2. Run the installer. If you see "Windows protected your PC", click "More info", then "Run anyway".
3. Choose where to install. The default is `%LOCALAPPDATA%\Programs\MIA` and does not need administrator rights. Pick an empty folder used only for MIA.
4. Open MIA from the Start menu or the desktop shortcut.

### Mac

1. Download `MIA-<version>-macos-arm64.dmg` and double-click it.
2. Drag **MIA.app** onto the **Applications** folder next to it.
3. Open MIA from Applications. Do not run it from inside the disk image; automatic updates will not work there.
4. If macOS says the developer cannot be verified, open System Settings > Privacy & Security, scroll down to MIA and click "Open Anyway".

### Updating

Click "Check for updates…" under General > Language & updates. An update replaces only the app; your settings, login and recordings are kept.

## Getting started

When MIA opens, its side window appears on the right of the screen. All three features start switched off.

To switch the interface to English, go to General > Language & updates, pick English and restart MIA.

**First choose Web or Steam under General > Game version.** If this is wrong, MIA will say it cannot find the game window.

### Web version

1. Click **Start game** at the bottom left of the side window. MIA opens a bundled browser with Mahjong Soul.
2. Log in to Mahjong Soul in that browser. The login is saved on your computer, so you only need to do this once.
3. Once you are in a game, turn on the features you want: AI advice, Shanten analysis, Deal-in analysis.

### Steam version

1. Start Mahjong Soul from Steam yourself. The Start game button is disabled for Steam.
2. Choose **Steam** under General > Game version.
3. On a Mac in full screen, the layout is already measured and works as is. For other window sizes, select your hand once: in a game, while you have **no called melds**, go to Shanten analysis > Settings > Select hand area and drag a rough box around your hand. MIA aligns it to the tiles and remembers it.
4. Turn on the features you want. On Steam, all three features work from screen recognition.
5. On a Mac in full screen, the game is on its own desktop and you cannot see the side window. Check **Show overlay on the game** under General to draw the advice over the table.

### Screen Recording permission on Mac

The first time you turn on a feature that reads the screen, macOS asks for Screen Recording permission. Allow **MIA** in System Settings > Privacy & Security > Screen Recording, then **quit MIA completely and open it again** for the permission to take effect.

### Common settings

| Where | What it does |
|---|---|
| AI advice > Settings > Play style | Switch to another AI style. Advice pauses for about half a second while it switches. |
| AI advice > Settings > Compare with | Run a second style side by side. Uses twice the computation and memory. |
| Each feature > Settings > Recognition method | Template matching, YOLO or Hybrid. All three feature pages share this one setting. |
| General > YOLO compute device | Run YOLO on the CPU or GPU. |
| General > Language & updates | Interface language (takes effect after a restart), check for updates, clear all data. |

## Things to know

### Mahjong Soul terms of service

- Third-party assistance tools may violate Mahjong Soul's terms of service. MIA's open-source license only covers how the code may be shared; **it does not mean you may use it with the game**.
- Any warning, suspension or other action against your account is your own risk.
- This project is for academic research. Use it for spectating, replays or games against bots, **not in ranked games**.

### Advice is only a reference

- Both the AI advice and the screen recognition can be wrong. When MIA is not sure about the game state, it holds back advice and explains why at the bottom of the window, but it cannot catch every mistake.
- Deal-in analysis only lists what follows from the rules. A tile that is not listed is not necessarily dangerous, and a listed tile is not guaranteed to be safe in every case.

### Use the default game appearance

- Template matching and YOLO were built from Mahjong Soul's **default tiles, table and tile backs**. With other cosmetics, recognition may become unreliable or miss tiles entirely.
- MIA reminds you once when you switch to YOLO or Hybrid.

### Performance

- On a Mac, YOLO runs on the Apple GPU.
- The Windows installer currently runs YOLO on the CPU only; GPU acceleration packages are not available yet. Template matching is the lighter option.

### Privacy and your data

MIA does not send your data anywhere. Your settings, Mahjong Soul login, recordings and logs stay on your computer:

| | Windows | Mac |
|---|---|---|
| Settings, login, recordings | `userdata\` in the install folder | `~/Library/Application Support/MIA` |
| Cache | `userdata\cache\` | `~/Library/Caches/MIA` |
| Logs | `userdata\logs\` | `~/Library/Logs/MIA` |

### Uninstalling

- **Windows**: open Settings > Apps > Installed apps, find MIA and click Uninstall. You will be asked whether to also delete your settings, login, recordings and logs.
- **Mac**:
  1. To delete your data as well, first click "Clear all data…" under General > Language & updates in MIA. MIA quits and deletes the three folders listed above.
  2. Move MIA.app from Applications to the Trash.
  3. In System Settings > Privacy & Security > Screen Recording, select MIA and click "−".

## FAQ

**I turned a feature on but no advice appears.**
Check the status message at the bottom of the side window. It says what MIA is waiting for, for example "Waiting for a game to start". You can also open the Event log under General to see which game events MIA has read.

**MIA says it cannot find the game window.**
Make sure General > Game version is set correctly. For the web version, you must play in the browser opened by Start game, not in your own browser.

**On a Mac, the Screen Recording prompt keeps appearing.**
After allowing MIA in System Settings, quit MIA completely and open it again. If MIA is not in the list, click "+" and add MIA.app from Applications.

**The Dock still shows an old icon on a Mac.**
Finder caches icons. Move MIA.app out of Applications and back, or log out and in again.

## Running from source

For people who want to change the code or build MIA themselves. You need Python 3.14 (main app), Python 3.12 (AI engine and YOLO) and Rust (to compile Mortal's libriichi).

```bash
git clone https://github.com/jjeyxb/mahjong.git
cd mahjong
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/fetch_models.py     # downloads the model weights and checks each SHA-256
.venv/bin/python tools/ui.py --live
```

The AI engine and YOLO each need their own Python 3.12 environment. The full steps, including the Windows commands, are in these documents (in Chinese):

- [docs/development.md](docs/development.md): installing from source, architecture and measurements from each stage (the previous README)
- [docs/handoff.md](docs/handoff.md): current state of the project and notes for whoever takes it over
- [docs/live.md](docs/live.md): live mode manual and status messages
- [docs/yolo.md](docs/yolo.md): the YOLO recognition environment
- [packaging/README.md](packaging/README.md), [packaging/windows/README.md](packaging/windows/README.md): packaging and releases

## About the project

- **Architecture**: the main app (PySide6) handles the interface and recognition. Mortal and YOLO each run in a separate process and talk to it over the MJAI protocol. The three features do not depend on each other, so a problem in one does not affect the others.
- **Play-style fine-tuning**: today's mahjong AIs play in very similar ways. This project took high-level players' game logs, picked players from both ends of two style axes (call rate and riichi rate), and fine-tuned four models. The styles carried over: the models reproduce 81% of the call-rate gap and 98% of the riichi-rate gap, with no loss in average placement.
- **Deal-in analysis**: instead of deal-in rates estimated from game statistics, it only reports what can be deduced from the rules, and every entry can say why.

Design rationale and full results are in [docs/development.md](docs/development.md) and [docs/decisions.md](docs/decisions.md) (in Chinese).

## License and credits

MIA is licensed under [AGPL-3.0-or-later](LICENSE). The licenses of third-party work and how this project uses it are listed in [NOTICE](NOTICE).

- **AI engine**: [Mortal](https://github.com/Equim-chan/Mortal) by Equim-chan (AGPL-3.0).
- **Base model**: [VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) (AGPL-3.0).
- **Style models**: [jjeyxb/mia-mortal-style](https://huggingface.co/jjeyxb/mia-mortal-style), fine-tuned in this project, labeled AGPL-3.0.
- **YOLO recognition models**: [jjeyxb/mia-yolo](https://huggingface.co/jjeyxb/mia-yolo), trained with Ultralytics by a project team member, labeled AGPL-3.0.

Mahjong Soul belongs to its rights holders. This project is not affiliated with Mahjong Soul or its operators.
