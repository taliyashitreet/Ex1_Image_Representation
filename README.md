# Ex1_Image_Representation
this exercise belong to image processing Course under the Artificial Intelligence track.<br />
In ex1.utils you will find an image processing like:

1. repressentation of RGB or Gray Scale image <br />
   the mathods imReadAndConvert() and imDisplay() using Opencv2, matplotlib  <br />
2. transform RGB image to YIQ and revers <br />
    the methods transformRGB2YIQ() and transformYIQ2RGB() using the fact: ![image](https://user-images.githubusercontent.com/77111035/160815881-3d364efe-e511-4920-93ff-a0941eb39749.png)

3. filter : Histogram equality  <br />
   the method hsitogramEqualize() - algorithm that calculate the cumsum of the image and change it Linear as much as possible for Equale histogram
4. filter : Quantization <br />
 the method quantizeImage() - create a new image with input number of colors  
 from this image: <br />
![image](https://user-images.githubusercontent.com/77111035/160814155-8b08878a-36f5-4321-9667-92f6cad05fe7.png)<br />
to this : <br />
![image](https://user-images.githubusercontent.com/77111035/160814193-d2bf505c-c311-4e9b-95c8-c38c75695682.png)<br />
In gamma you will find :
4. filter : Gamma correction on GUI 
 <img width="452" alt="image" src="https://user-images.githubusercontent.com/77111035/160812796-c027613c-28e4-4789-90c5-47a359f5d275.png">
