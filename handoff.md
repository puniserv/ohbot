# Handoff: Ohbot / OhBrain V3.1 not detected over USB

Repo: /Volumes/projects/priv/ohbot (empty git repo, no commits; only `ohbot.py/` empty dir, `warranty-claim.md`, this file).
User communicates in Polish.

## Original goal
Use Ohbot 2 from macOS (Apple Silicon, Python 3.14 system; plan: `uv venv -p 3.11` + `pip install ohbot`; library ohbot/ohbot-python supports Mac; auto port detection looks for "USB Serial Device" so on Mac pass `ohbot.init("/dev/cu.usbmodem...")`). Blocked: robot never enumerates.

## Hardware
- Board: OhBrain V3.1, MCU RP2040 (U2), crystal Y1 (12 MHz), regulator U1 (SOT-223, top-right of D1/F1), flash = small dark USON/WSON chip between J6 and C16 near R12/R13.
- SWD header labelled C D V G (SWCLK, SWDIO, V, GND). V pin measures 5 V by design.
- J1 = 2-pin header with blue jumper (function unknown). J3 = 3 empty through-holes. J7 = 2 small pads near R12/R13 (function unknown).
- Bottom side: through-hole only, no SMD, no visible damage.

## Symptoms
- Worked over USB before; stopped after being stored in its box.
- On plug-in: blue LED (D4) on, servos twitch once, head does not centre.
- No USB device at all on macOS (system_profiler, /dev/cu.*, /Volumes) or Windows. Several data-capable USB-C cables, both plug orientations, direct connection.
- Not even RPI-RP2 bootloader drive.

## Measurements done
- U1 output (tab): 3.3 V OK.
- R16, R17: ~5 kOhm each, one pad to GND -> CC pull-downs (5.1k), OK. (Earlier "3 ohm" reading was a mismeasurement.)
- Z1, Z2: one pad to GND each (ESD diodes on D+/D-); diode-mode readings symmetrical ("looks ok"). No short/open.
- USB-C shield to GND: continuity, normal.
- J7 pads: ~7 MOhm between them (open). Mapping to GND/R12/R13 NOT measured.
- Bridged J7 while plugging in (BOOTSEL attempt): nothing enumerated. J1 jumper removal test: never confirmed by user.

## Current hypotheses (most -> least likely)
1. Crystal Y1 dead / cracked joint (no 12 MHz -> no USB, even in bootrom).
2. Corrupted firmware in flash hangs before USB init (BOOTSEL pad unknown, J7 unconfirmed).
3. RP2040 USB peripheral damaged (ESD).

## Next steps
- Get from Ohbot (ohbot.co.uk): schematic / BOOTSEL location for V3.1, firmware .uf2, or replacement.
- SWD probe: Raspberry Pi Pico with debugprobe firmware (or RPi Debug Probe) -> C/D/G, robot powered by its own cable, `openocd -f interface/cmsis-dap.cfg -f target/rp2040.cfg`. Chip responds -> reflash via SWD, check clocks. No response -> RP2040/1.1V core issue.
- Don't solder while warranty claim is pending.

## Deliverables already produced
- `warranty-claim.md` (English email to Ohbot, placeholders: order number, date, name/address).
- Polish summary for a colleague (given in chat; hypotheses as above).
- Repair cost estimate (from knowledge, not current prices): parts 30-60 PLN best case (Pico probe 20-25, crystal 5-10, consumables 40-70 if missing), 60-120 PLN worst (RP2040 10-20) + hot air; labour ~100-150 PLN diag+reflash, 120-200 crystal, 250-400 RP2040 swap. Ceiling = price of new OhBrain board incl. UK shipping. Firmware file is prerequisite for any chip/flash swap.
