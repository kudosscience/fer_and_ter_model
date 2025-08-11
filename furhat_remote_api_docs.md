# Remote API - Furhat Developer Docs
The Furhat Remote API is one of three ways to program the robot. The other two being [blockly](../blockly) and [Kotlin Skill API](../skills).

Furhat Remote API is a way to connect and give commands to your Furhat robot from a program running on an external computer on the same network. A number of programming languages are supported, including Python, C#, JavaScript, Rust etc. (50+ languages, for full list see [Swagger documentation](https://swagger.io/docs/open-source-tools/swagger-codegen/))

We've created a [wrapper specifically for Python](about:blank/remote-api/#python-remote-api) to make it as easy as possible to start using it in a python application.

An intro, overview and setup instructions to the Remote API and the python wrapper is available in this video.

Requests for more commands to include in the Remote API can be sent to: [tech@furhatrobotics.com](mailto:tech@furhatrobotics.com)

Setup
-----

### Run the server on the Robot

The Remote API is included in Standard and Premium packages of the Furhat Robot. It can be started from the web interface of the robot, or in the SDK Launcher. When the API is running, the robot will start a Swagger Kotlin server listening on port 54321.

![](../images/remote_API_ScreenCapture.png)

![](../images/Furhat_Studio_RemoteAPI_ScreenShot.png)

### Testing the remote API using Postman

If you want, you can test the Remote API with the tool Postman:

1.  [Download Postman](https://www.postman.com/downloads/),
2.  Download and open the [Furhat Remote API yaml specification](https://furhat-files.s3-eu-west-1.amazonaws.com/other/furhat-remote-api.yaml),
3.  Test the different requests (look for API documentation below).

### Create a client for your specific programming language

In this step you will generate the code in your preferred programming language that will enable your program to communicate with Furhat.

If you want to to use the Remote API from Python, we recommend to use our pre-built [Python PyPi library](#python-remote-api), which wraps the Remote API and makes it easier to install and use.

1.  [Download Swagger](https://swagger.io/tools/swagger-ui/download/) or use their [online editor](https://editor.swagger.io/),
2.  Paste the content from [Furhat Client yaml file](https://furhat-files.s3-eu-west-1.amazonaws.com/other/furhat-remote-api.yaml) ,
3.  Click 'Generate Client' and select your language,
4.  Incorporate the client to your system and send requests to control the robot.

API Documentation
-----------------

### API Endpoints

All URIs are relative to _http://__IP of the robot__:54321_



* Endpoint: /furhat/attend
  * HTTP request method: POST
  * Description: Attend a user/location
* Endpoint: /furhat/face
  * HTTP request method: POST
  * Description: Change the character and mask, or texture and model (deprecated)
* Endpoint: /furhat/visibility
  * HTTP request method: POST
  * Description: Fade in or out the face (FaceCore only)
* Endpoint: /furhat/gesture
  * HTTP request method: POST
  * Description: Perform a gesture
* Endpoint: /furhat/led
  * HTTP request method: POST
  * Description: Change the colour of the LED strip
* Endpoint: /furhat/listen
  * HTTP request method: GET
  * Description: Make the robot listen, and get speech results
* Endpoint: /furhat/listen/stop
  * HTTP request method: POST
  * Description: Make the robot stop listening
* Endpoint: /furhat/say
  * HTTP request method: POST
  * Description: Make the robot speak
* Endpoint: /furhat/say/stop
  * HTTP request method: POST
  * Description: Make the robot stop talking
* Endpoint: /furhat/voice
  * HTTP request method: POST
  * Description: Set the voice of the robot
* Endpoint: /furhat/gestures
  * HTTP request method: GET
  * Description: Get all gestures
* Endpoint: /furhat
  * HTTP request method: GET
  * Description: Test connection
* Endpoint: /furhat/users
  * HTTP request method: GET
  * Description: Get current users
* Endpoint: /furhat/voices
  * HTTP request method: GET
  * Description: Get all the voices on the robot


### Authentication

None of the endpoints require authentication. Keep in mind that while this makes it easy to use, anyone on a larger network could theoretically listen in. It is recommended to stop the Remote API skill (or turn off the robot) when you are not actively working with it.

Models
------

*   [io.swagger.server.models.BasicParam](#basicparam)
*   [io.swagger.server.models.Frame](#frame)
*   [io.swagger.server.models.Gesture](#gesture)
*   [io.swagger.server.models.GestureDefinition](#gesturedefinition)
*   [io.swagger.server.models.Location](#location)
*   [io.swagger.server.models.Rotation](#rotation)
*   [io.swagger.server.models.Status](#status)
*   [io.swagger.server.models.User](#user)
*   [io.swagger.server.models.Voice](#voice)

/furhat
-------

##### GET

###### Summary:

Test connection

###### Description:

Used to verify if the server is running, return "hello world" upon success

###### Responses


|Code|Description  |
|----|-------------|
|200 |Status update|


/furhat/gestures
----------------

##### GET

###### Summary:

Get all gestures

###### Description:

Returns a JSON array with all gestures on the system (names + duration).

###### Responses


|Code|Description                |Schema     |
|----|---------------------------|-----------|
|200 |A list of possible gestures|[ Gesture ]|


/furhat/voice
-------------

##### POST

###### Summary:

Set the voice of the robot

###### Description:

Sets the voice of the robot using the name of the voice, can be requested by doing a GET request on this endpoint.

###### Parameters


|Name|Located in|Description          |Required|Schema|
|----|----------|---------------------|--------|------|
|name|query     |The name of the voice|Yes     |string|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/voices
--------------

##### GET

###### Summary:

Get all the voices on the robot

###### Description:

Returns a JSON array with voice names + languages.

###### Responses


|Code|Description|Schema   |
|----|-----------|---------|
|200 |Success    |[ Voice ]|


/furhat/users
-------------

##### GET

###### Summary:

Get current users

###### Description:

Get all current users (max: 2). Returns a JSON array containg Users (Rotation, Location, id).

###### Responses


|Code|Description         |Schema  |
|----|--------------------|--------|
|200 |successful operation|[ User ]|


/furhat/say
-----------

##### POST

###### Summary:

Make the robot speak

###### Description:

Makes the robot speak by either using text, or a URL (linking to a.wav file). If generatelipsync=true, it uses a .pho file hosted on the same url, or generates phonemes by itself.

**Note :** Lipsync does not work for local audio.

###### Parameters



* Name: text
  * Located in: query
  * Description: A string depicting a utterance the robot should say.
  * Required: No
  * Schema: string
* Name: url
  * Located in: query
  * Description: A url link to a audio file (.wav)
  * Required: No
  * Schema: string
* Name: blocking
  * Located in: query
  * Description: Whether to block execution before completion
  * Required: No
  * Schema: boolean
* Name: lipsync
  * Located in: query
  * Description: If a URL is provided, indicate if lipsync files should be generated/looked for.
  * Required: No
  * Schema: boolean
* Name: abort
  * Located in: query
  * Description: Stops the current speech of the robot.
  * Required: No
  * Schema: boolean


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |successful operation|Status|


/furhat/attend
--------------

##### POST

###### Summary:

Attend a user/location

###### Description:

Provides 3 modes of attention. 1. Attend user based on enum (CLOSEST, OTHER or RANDOM) 2. Attend user based on it's id (can be retrieved by using /furhat/users) 3. Attend location based on coordinates (x,y,z)

###### Parameters



* Name: user
  * Located in: query
  * Description: Make furhat attend a user. Example 'CLOSEST'
  * Required: No
  * Schema: [  ]
* Name: userid
  * Located in: query
  * Description: Make furhat attend specified user
  * Required: No
  * Schema: string
* Name: location
  * Located in: query
  * Description: Make furhat attend location, usage: x,y,z. Example -20.0,-5.0,23.0
  * Required: No
  * Schema: string


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/face
------------

##### POST

###### Summary:

Changes the appearance of the robot, using either character and mask, or texture and model.

###### Description:

Changes both the character and mask ([FaceCore face engine](../facecore/)), or texture and model (deprecated [OpenSceneGraph face engine](../textures/)), based on the mask/character or model/texture name. Case sensitive. Names can be retrieved from the web interface.

###### Parameters


|Name     |Located in|Description                      |Required|Schema|
|---------|----------|---------------------------------|--------|------|
|mask     |query     |Change the mask of the robot     |No      |string|
|character|query     |Change the character of the robot|No      |string|
|model    |query     |Change the model of the robot    |No      |string|
|texture  |query     |Change the texture of the robot  |No      |string|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/visibility
------------------

##### POST

###### Summary:

Fade in/out the face

###### Description:

Triggers an animation which fades the face out to black, or in again, with a set duration in the range of \[0,10000\] ms. Invalid input values will be coerced in, and missing durations will be set to the default value of 2000 ms. This command is only applicable to the [FaceCore face engine](../facecore/).

###### Parameters


|Name    |Located in|Description                                   |Required|Schema |
|--------|----------|----------------------------------------------|--------|-------|
|visible |query     |Whether the face should be made visible or not|Yes     |boolean|
|duration|query     |Duration of the fade animation in milliseconds|No      |integer|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/gesture
---------------

##### POST

###### Summary:

Perform a gesture

###### Description:

Performs a gesture based on 1. Gesture name (retrieve by GET request to /furhat/gestures) 2. Gesture definition, see example

###### Parameters


|Name    |Located in|Description                                 |Required|Schema           |
|--------|----------|--------------------------------------------|--------|-----------------|
|name    |query     |The gesture to do                           |No      |string           |
|blocking|query     |Whether to block execution before completion|No      |boolean          |
|body    |body      |Definition of the gesture                   |No      |GestureDefinition|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|
|400 |Parameters are wrong|Status|


/furhat/listen
--------------

##### GET

###### Summary:

Make the robot listen, and get speech results

###### Description:

Blocking call to get user speech input, language defaults to english\_US. Language parameter can be used to provide a different language. Return values can be found in the Status object as message and can be: - User speech - SILENCE - INTERRUPTED - FAILED

###### Parameters


|Name    |Located in|Description                                  |Required|Schema|
|--------|----------|---------------------------------------------|--------|------|
|language|query     |The language to listen for, defaults to en-US|No      |string|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/listen/stop
-------------------

##### POST

###### Summary:

Make the robot stop listening

###### Description:

Aborts the listen

###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/led
-----------

##### POST

###### Summary:

Change the colour of the LED strip

###### Description:

Changes the LED strip of the robot, colours can be between 0-255 (above 255 is changed to 255). Any parameter not provided defaults to 0.

###### Parameters


|Name |Located in|Description        |Required|Schema |
|-----|----------|-------------------|--------|-------|
|red  |query     |The amount of red  |No      |integer|
|green|query     |The amount of green|No      |integer|
|blue |query     |The amount of blue |No      |integer|


###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


/furhat/say/stop
----------------

##### POST

###### Summary:

Make the robot stop talking

###### Description:

Stops the current speech.

###### Responses


|Code|Description         |Schema|
|----|--------------------|------|
|200 |Successful operation|Status|


Models
------

### Status


|Name   |Type   |Description|Required|
|-------|-------|-----------|--------|
|success|boolean|           |No      |
|message|string |           |No      |


### Gesture


|Name    |Type            |Description|Required|
|--------|----------------|-----------|--------|
|name    |string          |           |No      |
|duration|integer (double)|           |No      |


### User


|Name    |Type    |Description|Required|
|--------|--------|-----------|--------|
|id      |string  |           |No      |
|rotation|Rotation|           |No      |
|location|Location|           |No      |


### Location


|Name|Type  |Description    |Required|
|----|------|---------------|--------|
|x   |double|Robot's left   |No      |
|y   |double|Robot's forward|No      |
|z   |double|Robot's up     |No      |


### Voice


|Name    |Type  |Description|Required|
|--------|------|-----------|--------|
|name    |string|           |No      |
|language|string|           |No      |


### Rotation


|Name|Type  |Description|Required|
|----|------|-----------|--------|
|x   |double|           |No      |
|y   |double|           |No      |
|z   |double|           |No      |


### GestureDefinition

The class name needs to be **furhatos.gestures.Gesture** otherwise it won't be parsed as a gesture. Examples can be found below.


|Name  |Type     |Description|Required|
|------|---------|-----------|--------|
|name  |string   |           |No      |
|frames|[ Frame ]|           |No      |
|class |string   |           |No      |


#### GestureDefinition examples

All BasicParams are listed here:

```
    //All parameters have values between 0.0 and 1.0 (Except for the ones at the bottom).
    EXPR_ANGER
    EXPR_DISGUST
    EXPR_FEAR
    EXPR_SAD
    SMILE_CLOSED
    SMILE_OPEN
    SURPRISE
    BLINK_LEFT
    BLINK_RIGHT
    BROW_DOWN_LEFT
    BROW_DOWN_RIGHT
    BROW_IN_LEFT
    BROW_IN_RIGHT
    BROW_UP_LEFT
    BROW_UP_RIGHT

    EYE_SQUINT_LEFT
    EYE_SQUINT_RIGHT

    LOOK_DOWN
    LOOK_LEFT
    LOOK_RIGHT
    LOOK_UP

    PHONE_AAH
    PHONE_B_M_P
    PHONE_BIGAAH
    PHONE_CH_J_SH
    PHONE_D_S_T
    PHONE_EE
    PHONE_EH
    PHONE_F_V
    PHONE_I
    PHONE_K
    PHONE_N
    PHONE_OH
    PHONE_OOH_Q
    PHONE_R
    PHONE_TH
    PHONE_W

    LOOK_DOWN_LEFT
    LOOK_DOWN_RIGHT
    LOOK_LEFT_LEFT
    LOOK_LEFT_RIGHT
    LOOK_RIGHT_LEFT
    LOOK_RIGHT_RIGHT
    LOOK_UP_LEFT
    LOOK_UP_RIGHT

    //The following parameters have values in the range -50.0 to 50.0
    NECK_TILT
    NECK_PAN
    NECK_ROLL
    GAZE_PAN
    GAZE_TILT

```


Note that you can also use any of the [FaceCore](../facecore)\-compatible ARKitParams or CharParams, or any pre-recorded gestures, assuming the FaceCore face engine is used.

A couple of gesture examples:

Built-in BigSmile

```
{
  "name":"BigSmile",
  "frames":[
    {
      "time":[0.32,0.64],
      "persist":false, <- Optional
      "params":{
        "BROW_UP_LEFT":1,
        "BROW_UP_RIGHT":1,
        "SMILE_OPEN":0.4,
        "SMILE_CLOSED":0.7
        }
    },
    {
      "time":[0.96],
      "persist":false, <- Optional
      "params":{
        "reset":true
        }
    }],
  "class":"furhatos.gestures.Gesture"
}

```


Custom gesture

```
{
  "frames": [
    {
      "time": [
        0.17, 1.0, 6.0
      ],
      "params": {
        "NECK_ROLL": 25.0,
        "NECK_PAN": -12.0,
        "NECK_TILT": -25.0
      }
    },
    {
        "time": [
            7.0
        ],
        "params": { 
            "reset": true
        }
    }
  ],
  "name": "Cool Thing",
  "class": "furhatos.gestures.Gesture"
}

```


### Frame

A list of times can be provided, at those times the params will be executed.


|Name  |Type      |Description|Required|
|------|----------|-----------|--------|
|time  |[ double ]|           |No      |
|params|BasicParam|           |No      |


### BasicParam

All supported parameters can be found here [BasicParam](#gesturedefinition).


|Name      |Type  |Description|Required|
|----------|------|-----------|--------|
|BasicParam|object|           |        |


Python Remote API
-----------------

### Description

To simplify the use of the Furhat Remote API from Python, there is a package on PyPi called [furhat-remote-api](https://pypi.org/project/furhat-remote-api/).

### Installation

You can install the package using pip:

```
pip install furhat-remote-api

```


(you may need to run `pip` with root permission: `sudo pip install furhat-remote-api`)

### Usage

This shows how the different methods in the Remote API can be invoked from Python.

```
from furhat_remote_api import FurhatRemoteAPI

# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()

# Set the voice of the robot
furhat.set_voice(name='Matthew')

# Say "Hi there!"
furhat.say(text="Hi there!")

# Play an audio file (with lipsync automatically added) 
furhat.say(url="https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav", lipsync=True)

# Listen to user speech and return ASR result
result = furhat.listen()

# Perform a named gesture
furhat.gesture(name="BrowRaise")

# Perform a custom gesture
furhat.gesture(body={
    "frames": [
        {
            "time": [
                0.33
            ],
            "params": {
                "BLINK_LEFT": 1.0
            }
        },
        {
            "time": [
                0.67
            ],
            "params": {
                "reset": True
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    })

# Get the users detected by the robot 
users = furhat.get_users()

# Attend the user closest to the robot
furhat.attend(user="CLOSEST")

# Attend a user with a specific id
furhat.attend(userid="virtual-user-1")

# Attend a specific location (x,y,z)
furhat.attend(location="0.0,0.2,1.0")

# Set the LED lights
furhat.set_led(red=200, green=50, blue=50)

```
