# Unifi NVR Person Detector
This script is used to run YOLO object detection on video files recorded by the unifi camera and send notification if that happens.

This is accomplished by tailing the log file of the NVR remotely (using python-sshtail)


# Usage
- Clone Darknet `git clone https://github.com/AlexeyAB/darknet.git /opt/darknet`
- Build Darknet `cd /opt/darknet/ && make`
- Install Darknet
- Modify `settings.config` to add your NVR config details
- Create startscript with content from the videodetector.service
- Enable autostartup of script `sudo systemctl enable unifwatcher.service`
- Startup the service for the first time

Control the unifi watcher service using the following:
```
sudo systemctl enable unifwatcher.service   #Start upd service at startup
sudo systemctl start unifiwatcher.service    #Start upd service
sudo systemctl stop unifiwatcher.service     #Stop upd service
sudo systemctl status unifwatcher.service   #Show status of upd service
```



#Special thanks

Original concepts and inspiration taken from https://github.com/christian-ek/unifi-person-detector
