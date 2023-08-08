```meta
title: Sending Data With Android and USB
date: 2022/07/14
tags: android, usb
```
# Sending Data With Android and USB

Despite being a Linux user for 10 years and lesiurely reading books on OSes, I have never really had to look deep into any kernel topic beyond a basic level.
However, now I want to transmit data from my Android phone to a laptop at a high data-rate. This could be done over the LAN with WiFI, but would be subject to WiFi availability and speeds -- which on my router are not great. USB was an obvious choice and I looked forward to learning a little bit about it and its firmware in Linux.

What I want is to transfer a video feed and some sensor data over USB. The Android device should act not as the host, but as the device or 'gadget'. The host is the laptop/desktop. The Android SDK offers some high-level functionality for dealing with USB. Unforunately it is mostly for when the Android device is the host.
So I spent some time seeing how I can accomplish this. The most general way to approach it is to write a USB gadget driver for the Android device and load it into the kernel. But this is pretty complicated, and requires rooting your phone which was a no go for me. So my kernel aspirations are still on hold. What I want is a userland application (an Android application, possible with C++ code using the NDK) to do all of the USB configuration and transfers.
`libusb` is a popular library I had heard of, but it only operates on the host. On Linux, it [appears](https://github.com/libusb/libusb/blob/master/libusb/os/linux_usbfs.c) to rely on `usbfs`, which is only controlling the host. Android has high-level support for "accessories" but this requires following some protocol with not much documentation, and appears to be targeted for wearables and such.

I know the Android SDK includes the `adb` tool which is a bridge that allows you to interact with the phone from a computer and execute shell commands, data transfers, etc. through a USB connection. So `adb` was my starting point. Digging into the code, it uses Linux's `ConfigFS` and `FunctionFS`. `FunctionFS` is also not detailed very well, but it offers a way for userland code to add a new USB endpoint and facilitate transfers. This is exactly what I needed!

There is not great documentation on configfs/functionfs, but luckily the `adb` daemon provides a good example of using it. So I looked at the ADB code and it didn't look like too much work, thanks to the nice [Android source code browser](https://cs.android.com/android/platform/superproject/+/master:packages/modules/adb/daemon/usb_ffs.cpp;bpv=1;bpt=1;l=252). But using configfs/functionfs requires creating files in the `/sys/` or `/dev/` directories, which you can't do on Android unless you root the device. So this turned out to be a dead end.

Next I stumbled on to the USB tethering feature. This is meant for using your phone's mobile internet on a laptop. But it creates a LAN between your android and other device, so you can also just use it to send IP packets between them. I was able to easily send data between the devices with USB tethering on! I also tried sending UDP packets continuously, and it only dropped a very low percentage of them. Next up: sending a video stream from the phone's camera.

## Sending the Video via Tethered USB Network

The Android docs on using a device's camera were a little murky. Or maybe I was just impatient and didn't want to read more than I had to. There are several APIs at different levels of generality/ease-of-use. I think you can get the raw camera buffers and apply a codec in your own code, for example. But I wanted to use the `MediaRecorder` class because it implements the `mpegts` container format I want to use for transmitting in real-time. The class has a method `setOutputFile` that you can pass with a string filename, or a java FileDescriptor. So I figured I could open my socket and pass it as the fd. Well... no, this caused a crash with error code `-38` in the `MediaRecorder::prepare()` call.
After an hour of reviewing Android source code, I believe the crash is coming from [here](https://cs.android.com/android/platform/superproject/+/master:frameworks/av/media/libmediaplayerservice/StagefrightRecorder.cpp;drc=798331bb6d9e88c8e8bbf825544f4ddb96b940b3;bpv=1;bpt=1;l=411?gsn=setOutputFile&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dframeworks%2Fav%2Fmedia%2Flibmediaplayerservice%2FStagefrightRecorder.cpp%23BgUA4AwS7v7SOqBdyx3kjupAHOkiD46ONnTRggHcWg0&gs=kythe%3A%2F%2Fandroid.googlesource.com%2Fplatform%2Fsuperproject%3Flang%3Dc%252B%252B%3Fpath%3Dframeworks%2Fav%2Fmedia%2Flibmediaplayerservice%2FStagefrightRecorder.h%23aD9TAExpI8lkHqClZzHo9AN_grWrEgOFR0omcDetBXY). Specifically, I think that calling `ftruncate` on the socket is causing the issue. I thought I could hack this by using `dup2` to overwrite the fd after calling `setOutputFile` and before `prepare`, but actually the code seems to run in a different process as a service and interacts via RPC, so this trick didn't work. I've never had to do it, but I wondered how the app process could send the fd (which is just an integer) to the daemon. Apparently on Linux you process can send file descriptors to another via Unix domain sockets and the `sendmsg`/`recvmsg` with ancillary data.

So I needed to write to a file descriptor backing `ftruncate`. My next thought was to use a fifo/named-pipe. I don't know if the POSIX C `mkfifo` is permitted on Android, but there is the `ParcelFileDescriptor.makePipe` that was easy enough to use. I used the write end as the `setOutputFile` argument and spawned a thread to send UDP packets from the read end. And ... it works! I can ffplay the resulting mpegts stream on my desktop and see it. The stream is pretty good over UDP. Sometimes packets are dropped in a bunch, but it's good enough for me.

###### ConfigFS/FunctionFS Notes (which I didn't wind up using)
 - [Not totally relevant video](https://www.youtube.com/watch?v=mQYh4xYG5a4)
	- A `function` is the implementation of an interface in the kernel
	- FunctionFS delegates USB control to userland
	- ConfigFS is an interface that allows modularly mixing/matching USB gadgets. Allows creaton/deletion of kernel objects by user (in shell)
	- The user specifies a USB 'scheme' (its vendor, product, function+endpoints)
