# Camera and Audio feeds - Furhat Developer Docs
With our 1.20.0 update our system allows for the user to access the camera feed from our built-in camera. The audio feed is also available since 2.6.0. The raw feeds can be accessed either by the skill or from an external computer.

> **Note**: The SDK only supports the audio feed, we don't support computer cameras or other video feeds. On the robot, the feeds are only exposed for thoses of type ['Research'](about:blank/robot/#robot-type).

Enabling/Disabling
------------------

First the feeds have to be enabled, this can be done in two ways. Either programmatically inside a skill, or by using the web-interface.

Programmatically, the feed can be enabled by using the furhat object (available inside a state):

```
val example = State {

  onEntry {
    /** Camera feed */
    furhat.cameraFeed.enable()     //Enables the camera feed
    furhat.cameraFeed.disable()    //Disables the camera feed
    furhat.cameraFeed.isOpen()     //Returns a boolean depicting if the feed is open
    furhat.cameraFeed.isClosed()   //Returns a boolean depicting if the feed is closed.
    furhat.cameraFeed.port()       //Returns the port of the camera feed as an Int.

    /** Audio feed */
    furhat.audioFeed.enable()     //Enables the audio feed
    furhat.audioFeed.disable()    //Disables the audio feed
    furhat.audioFeed.isOpen()     //Returns a boolean depicting if the feed is open
    furhat.audioFeed.port()       //Returns the port of the audio feed as an Int.
  }
}

```


Note that the feeds can only be enabled programmatically from a skill running on the robot, not if you are running the skill from IntelliJ. The external feeds can also be enabled/disabled by the click of a button in the web-interface. They can be found under Settings > External Feeds.

Accessing the Camera feed from the skill
----------------------------------------

You can access the camera feed from the skill by either requesting a snapshot, or by adding a listener to the feed. You can get access to both the images being streamed from the camera, as well as information about the faces being detected in the image (with 2D coordinates).

This is how you retrieve a snapshot from the camera (which will also automatically enable the camera feed):

```
val (image,faces) = furhat.cameraFeed.getSnapShot()

```


This function will return the `image` as a `BufferedImage` and the `faces` as a list of `FaceData` objects, which contain information about 2D coordinates for each face.

Note that these methods will block until the image is retrieved. If the camera feed is not successfully enabled, a null will be returned within a certain timeout (5000ms per default, but you can provide another timeout).

You can also get a camera snapshot with a `FaceData` object related to a specific user:

```
val (image,face) = users.current.getSnapShot()

```


Another option is to add a listener to the camera feed:

```
object MyCameraFeedListener: CameraFeedListener() {
    override fun cameraImage(image: BufferedImage, imageData: ByteArray, faces: List<FaceData>) {
        // You can choose whether to process the image as raw JPEG data or as a BufferedImage.
    }
}
// Add your listener to the camera feed. This will automatically enable the feed. Note that it can only be enabled programmatically when the skill is running on the robot. Otherwise, you have to enable it manually from the web interface.
furhat.cameraFeed.addListener(MyCameraFeedListener)

```


Accessing the Audio feed from the skill
---------------------------------------

To access the audio feed from the skill, you can create an `AudioFeedListener`:

```
object MyAudioFeedListener: AudioFeedListener() {
    override fun audioData(data: ByteArray) {
        // One frame of audio data.
        // The audio data is encoded as signed, 16 bit, little-endian linear PCM format, 16KHz, stereo
    }
}
// Add your listener to the audio feed. This will automatically enable the feed. Note that it can only be enabled programmatically when the skill is running on the robot. Otherwise, you have to enable it manually from the web interface.
furhat.audioFeed.addListener(MyAudioFeedListener)

```


The audio is provided in stereo. The input audio (microphone) is in the left channel, and the output audio (speech synthesis) is in the right channel.

External access to the Camera feed
----------------------------------

You can also access the camera feed from an external computer, for example running a python script.

When the camera feed is enabled, a url will be provided. The url leads to a `ZMQ.SUB` socket where a stream of images and metadata is published. After each JPEG image (binary string) follows a JSON-formatted object containing annotations of that image. The annotation object contains a `timestamp` (unix epoch time) and a `users` array, containing information about each of the detected users. For each user, the following information is provided:

```
{
      "id":1,
      "bbox":{
        "x":..,
        "y":..,
        "w":...,
        "h":...
      },
      "landmarks":[x0,y0,x1,y1,x2,y2...],
      "pos":{
        "x":x,
        "y":y,
        "z":z
      },
      "rot":{
        "pitch":p,
        "yaw":yaw,
        "roll":roll
      },
      "emotion":{
        "hap":x,
        "sad":y,
        "ang":z,
        "sur":w,
        "neu":n
      },
      "faceprint":[...]
}

```


Additionaly, a timestamp is added to the the JPEG image itself as EXIF data. To read the meta data out of the image you can use any library that reads EXIF data out of a jpeg image. The meta data is the User Comment with a json object that has a single field `timestamp` in unix epoch time. This can be used to ensure that the image and annotation data belongs together.

Example of how to connect a Furhat skill to an external computer for image processing and display of annotated camera images can be found [here](https://github.com/FurhatRobotics/tutorials/tree/main/camerafeed-demo) - see the [object recognition tutorial](../tutorials/camerafeed/).

External access to the Audio Feed
---------------------------------

When the audio feed is enabled, a url will be provided. The url leads to a `ZMQ.SUB` socket where an audio stream is published.

The audio is encoded as signed, 16 bit, little-endian linear PCM format, 16KHz, stereo (same format as WAV-files are encoded). The input audio (microphone) is in the left channel, and the output audio (speech synthesis) is in the right channel.

Example of how to use the Furhat audio feed can be found [here](https://github.com/FurhatRobotics/tutorials/tree/main/audiofeed-demo) - see the [audio streaming tutorial](../tutorials/audiofeed/).