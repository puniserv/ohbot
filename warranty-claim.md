Subject: Warranty claim – Ohbot with OhBrain V3.1 not detected over USB

Hello,

I would like to make a warranty claim for my Ohbot. The OhBrain board no longer shows up as a USB device on any computer.

Order number: [ORDER NUMBER]
Purchase date: [DATE]
Board: OhBrain V3.1 (RP2040)

What happened
The robot worked over USB before. After it had been stored in its box for a while, it stopped connecting. It has not been dropped or modified.

Symptoms
- When the cable is plugged in, the blue LED lights and the servos twitch once. Nothing happens after that, and the head does not centre.
- No USB device appears on the computer. It does not appear as a serial port, as an "Arduino Micro" / "USB Serial Device", or as the RP2040 bootloader drive (RPI-RP2).
- The behaviour is the same on a Mac (Apple Silicon, macOS) and on a Windows PC.
- I tried several USB-C cables, including ones I know carry data, in both plug orientations, connected directly to the computer without a hub.

Checks I have done (multimeter, cable unplugged unless stated otherwise)
- 3.3 V rail: correct, 3.3 V at the U1 regulator output with the board powered.
- USB-C CC pull-downs (R16, R17): about 5 kOhm to GND each, as expected.
- D+/D- protection diodes (Z1, Z2): diode-mode readings are symmetrical on both data lines. I found no short and no open circuit.
- USB-C connector: GND and shield continuity are fine. No visible damage.
- To try to force the bootloader, I briefly bridged the J7 pads while plugging in. No device appeared.

No soldering or component replacement has been done.

Conclusion
Power and the USB-C connector look fine. The fault seems to be in the RP2040, the 12 MHz crystal (Y1) or the flash/firmware: the board never enumerates, not even in bootloader mode.

I would be grateful for a repair or replacement of the OhBrain board. If you think a firmware reflash might fix it, please send the .uf2 file and instructions for entering BOOTSEL mode on V3.1.

I can send photos of both sides of the board.

Kind regards,
[NAME]
[ADDRESS / PHONE]
