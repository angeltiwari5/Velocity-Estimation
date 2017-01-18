Videos are captured from a static camera .(the aerial view is taken of a particular path).
 
1.VideoCapture cap("video.avi"); // open the video file for reading
2. bool bSuccess = cap.read(frame); // read a new frame from video
3.Background Subtraction algo implemented:
-MOG and MOG2 to create the foreground mask of the moving objects
4.All the foreground frames created using the above methods are saved
5.Now these frames are taken as input in the Optical flow .
-the intensities of pixel in consecutive frames is same where intensity is a function of(u,v,t) 
u is change in x coordinate,v is change in y coordinate,t is change in time thereby we got time using tvl1 method.
-a threshold of 60 frame interval is taken so as to check the difference in the flow of moving objects in the frames.
-it draws an optical flow window of the motion flow in the input frames.
-gives the time taken by the vehicles in those input interval frame.
6.Knowing the distance we get different velocities in all these intervals .
7.Now the average velocity is computed .