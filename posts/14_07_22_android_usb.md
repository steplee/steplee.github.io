# Android USB

Despite being a Linux user for 10+ years and lesiurely reading books on OSes, I have never really had to look deep into any kernel topic beyond an intermediate level.
However, now I want to transmit data from my Android phone to a laptop at a high data-rate. This could be done over the LAN, but would be subject to WiFi availability and speeds -- which on my router are not great.

What I want is to transfer a video feed and some sensor data over USB. The Android device should act not as the host, but as the 'gadget' as some people term it. The host is the laptop/desktop. The Android SDK offers some high-level functionality for dealing with USB. Unforunately it is mostly for when the Android device is the host.
So I spent some time seeing how I can accomplish this. The most general way to approach it is to write a USB gadget driver for the Android device and load it into the kernel. But this is pretty complicated. What we want is a userland application (and Android application, with C++ code using the NDK) to do all of the USB configuration and transfers.
`libusb` is a popular library I had heard of, but it only operates on the host. Android has high-level support for "accessories" but this requires following some protocol with not much documentation, and appears to be targeted for wearables and such.

I know the Android SDK includes the `adb` tool which is a bridge that allows you to interact with the phone from a computer and execute shell commands, data transfers, etc. through a USB connection. So `adb` was my starting point. Digging into the code, it uses Linux's `FunctionFS`. `FunctionFS` is also not detailed very well, but it offers a way for userland code to add a new USB endpoint and facilitate transfers. This is exactly what I needed!
The process goes like this:
 1. Application code creates a functionfs module and mounts it. "ep0" is a file that represents endpoint 0.
 2. Application code fills structs with metadata about the new endpoints and writes it to "ep0"
 3. The kernel functionfs module creates the new endpoint

There is not great documentation on functionfs, but luckily the `adb` daemon provides a good example of using it.
Adb is largely self-contained, but there appears the `UsbGadget` and `MonitorFfs` classes play a role.
The [`open_functionfs`](https://cs.android.com/android/platform/superproject/+/master:packages/modules/adb/daemon/usb_ffs.cpp;bpv=1;bpt=1;l=252?gsn=open_functionfs&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dpackages%2Fmodules%2Fadb%2Fdaemon%2Fusb_ffs.h%23wcPnFn6hhd6u3sUNwsclPIF4c5_j3nF6htwjeXm1Cxc&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dpackages%2Fmodules%2Fadb%2Fdaemon%2Fusb_ffs.cpp%23hAQE_jtK92ASwsz5TP6IMbNIY-HlCosIMSsHeHmeB7Q) opens the paths `/dev/usb-ffs/adb/{ep0,ep1,ep2}`, which seem to be created [here](https://cs.android.com/android/platform/superproject/+/master:hardware/interfaces/usb/gadget/1.2/default/lib/UsbGadgetUtils.cpp;bpv=1;bpt=1;l=191?gsn=addAdb&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dhardware%2Finterfaces%2Fusb%2Fgadget%2F1.2%2Fdefault%2Flib%2Finclude%2FUsbGadgetCommon.h%23N-KZXGkH_qBFEnMVKr5rViGdf6MSnkx5Vny_rd4sa9A&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dhardware%2Finterfaces%2Fusb%2Fgadget%2F1.2%2Fdefault%2Flib%2FUsbGadgetUtils.cpp%23ddz_kn0L6mgRwl0H1gpXa9h2Vmn1hTx75jd5bHAMfrM), which is ultimately invoked by usbd.cpp [usbhere](https://cs.android.com/android/platform/superproject/+/master:system/core/usbd/usbd.cpp;drc=fd8332d402a8cdc3dcb5436209a74f2c85747c66;l=47), which seems to invoked as a process by the actual OS `init()` process.

BUT, I'll need avoid all this crap...

## Sending via Tethered USB Network

I want to use the standard `MediaRecorder` class, but call `setOutputFile` with a socket instead of an actual storage file. This causes a crash with error code -38. After an hour of reviewing Android code, I believe the crash is coming from [here](https://cs.android.com/android/platform/superproject/+/master:frameworks/av/media/libmediaplayerservice/StagefrightRecorder.cpp;drc=798331bb6d9e88c8e8bbf825544f4ddb96b940b3;bpv=1;bpt=1;l=411?gsn=setOutputFile&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dframeworks%2Fav%2Fmedia%2Flibmediaplayerservice%2FStagefrightRecorder.cpp%23BgUA4AwS7v7SOqBdyx3kjupAHOkiD46ONnTRggHcWg0&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dframeworks%2Fav%2Fmedia%2Flibmediaplayerservice%2FStagefrightRecorder.h%23aD9TAExpI8lkHqClZzHo9AN_grWrEgOFR0omcDetBXY). Specifically, I think that calling `ftruncate` on the socket is causing the issue. I thought I could use `dup2` to overwrite the fd after calling `setOutputFile` and before `prepare`, but actually the code seems to run in a different process as a service and interacts via RPC, so this trick won't work. A process can send file descriptors to another via Unix domain sockets and the `sendmsg`/`recvmsg` with ancilary data.

So I need to write to a file descriptor backing `ftruncate`. My next thought was to use a fifo/named-pipe. I don't know if the POSIX C `mkfifo` is supported on Android, but there is the `ParcelFileDescriptor.makePipe` that was easy enough to use. I used the write end as the `setOutputFile` argument and spawned a thread to send UDP packets from the read end. And ... it works! I can fplay the resulting mpegts stream on my desktop with ffplay.

```python
def main():
	x = 2
	return 1
```

## Notes from [this video](https://www.youtube.com/watch?v=mQYh4xYG5a4)
 - A `function` is the implementation of an interface in the kernel
 - FunctionFS delegates USB control to userland
 - ConfigFS is an interface that allows modularly mixing/matching USB gadgets. Allows creaton/deletion of kernel objects by user (in shell)
