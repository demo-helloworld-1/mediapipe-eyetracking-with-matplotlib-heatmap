This Code is written using MediaPipe, 

use 
pip install cv
pip install mediapipe
pip install pyautogui

python main.py

For eye Calibration,

we have 2 calibrations required to start the eye tracking
C:
Complete face calibration,
once your web cam is turned on. look towards your left side of the screeen and press C, you will find the response saying left calibration successfull. 
now for Right Side of the screen and Press C
Now for Top of your Screeen and press C
Now for the Bottom

I:
Similarly, now dont move your face and just move your iris. if it didnt trigger, keep pressing until it records, slight face movement is negligable.
here also
Right Side
Left Side
Top 
Bottom


Once the Calibration is successful. you will be able to see a eye on your screen where you are looking

Press ESC to stop the Program

once the Program stops
now we can create the heatmap by the below command
python python generateHeatMap.py

A heat map will be generated.
