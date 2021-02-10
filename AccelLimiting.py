import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path'] = r'H:\Programs\FFmpeg\bin\ffmpeg.exe'
FFMpegWriter = anim.writers['ffmpeg']
Writer = anim.FFMpegWriter(fps=10)


#Generic Point Class
class Point(object):
    def __init__(self,x,y):
        self.X = x
        self.Y = y

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def setX(self, x):
        self.X = x

    def setY(self, y):
        self.Y = y

#Math algorithm to solve for a quadratic equation given 3 values
def calc_parabola_vertex(x1,y1,x2,y2,x3,y3):
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3)  + x1 * (y3-y2)) / denom
    B = (x3 *x3  * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C =  (x2 * x3 *  (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2  * (x1-x2) * y3) /  denom
    return A,B,C

#Determine which subject to look at
listOfSubjectNums = [1,4,5,6,7,8,10,11]
def getSubjectDataframe():
    subjectNum = input("Which Subject Would You Like to Review: ")
    if int(subjectNum) in listOfSubjectNums:
        return subjectNum
    else:
        print("The number you entered is not a valid subject number, please choose another")
        getSubjectDataframe()

currentSubject = getSubjectDataframe()
path = r'H:\Documents\HoustonData\On-road Driving Study\Quantitative Data'
pathB = r'H:\Documents\TungAlgorithm\Driver-Predictions\ord\data\processed\baseline\pp'
if(int(currentSubject) <= 10):
    currentDataFramePath = path + "\P0" + str(currentSubject) + ".csv"
    currentBaselinePath = pathB + "\P0" + str(currentSubject)  + ".csv"
else:
    currentDataFramePath = path + "\P" + str(currentSubject) + ".csv"
    currentBaselinePath = pathB + "\P" + str(currentSubject) + ".csv"
currentDataFrame = pd.read_csv(currentDataFramePath)
currentBaseline = pd.read_csv(currentBaselinePath)

#Grouping for Subjects taken from Tung's study
noAccelEffect = [1,4,8,10]
accelEffect = [5,6,7,11]

#Generate Histogram of pp
ppCol = currentDataFrame['pp_nr5']
baselinePP = currentBaseline['pp_nr2']
ppColOmit = ppCol.dropna()
baselinePPOmit = baselinePP.dropna()
currentThreshold = ppColOmit.mean()
ppColAdjusted = []

#Adjusts PP Data for Data
for time in range(ppCol.size):
    if(pd.isna(ppCol[time])):
        ppColAdjusted.append(ppCol[time - 1])
    else:
        ppColAdjusted.append(ppCol[time])

#plot of Histograms and Current Threshold
sns.distplot(baselinePPOmit,  hist=True, kde = True)
sns.distplot(ppColOmit, hist=True, kde = True)

plt.axvline(currentThreshold, color = 'red')
plt.legend(['Stress Threshold'])
plt.show()

#Function to Get Slope
def getSlope(A, B):
    return (B.getY() - A.getY())/(B.getX() - A.getX())


time = 6
timeArray = []
slopeArray = []

#Copy of Original Data to Be Used in Graphing
accelData = list(currentDataFrame['Accelerator'])

#Copy of Original Data to be Smoothed
accelDataLine = []
accelDataLine = list(currentDataFrame['Accelerator'])

#Determine Slope of Acceleration that on average causes stress
while time < currentDataFrame['Time'].size - 1:
    if (ppColAdjusted[time] > currentThreshold and abs(accelData[time - 5] - accelData[time]) > 5):
        previousPoint = Point(time-5, accelData[time-5])
        currentPoint = Point(time, accelData[time])
        slope = abs(getSlope(previousPoint, currentPoint))
        slopeArray.append(slope)
    time += 1

slopeAverage = sum(slopeArray)/len(slopeArray)

time = 6
startingPointArray = []


#Determines the intervals in which the car is speeding up/slowing down and recording points to do quadratic regression
while time < currentDataFrame['Time'].size - 1:
    if(ppColAdjusted[time] > currentThreshold and abs(accelData[time - 5] - accelData[time]) > 5):
        slopeTester = Point(time-5, accelData[time-5])
        timeArray.append(time)
        startingAccel = accelDataLine[time]
        startingPoint = Point(time, startingAccel)
        startingPointArray.append(startingPoint.getX())
        counter = time
        while(ppColAdjusted[counter] > currentThreshold and counter < len(ppColAdjusted) - 1):
            counter += 1
        endPoint = Point(counter, accelDataLine[counter])
        slope = getSlope(startingPoint, endPoint)
        if(abs(slope) + 1.75 < slopeAverage):
            time += 1
            while(time <= counter):
                accelDataLine[time] = accelDataLine[time - 1] + slope
                time += 1
        else:
            while(abs(getSlope(startingPoint, endPoint)) + 1.75 > slopeAverage and counter < currentDataFrame['Time'].size - 1):
                endPoint.setX(counter)
                endPoint.setY(accelDataLine[counter])
                counter += 1
            time += 1
            while (time <= counter):
                accelDataLine[time] = accelDataLine[time - 1] + getSlope(startingPoint, endPoint)
                time += 1
    else:
        time += 1

#Generates Max Value for Each Subject
rangeMax = len(ppColAdjusted) + 1

#Adjust's Time to Start after Start-up Zone
adjustedTime = []
for i in range (7,rangeMax):
    adjustedTime.append(i)

#Adjusts Original Acceleration Curve to Match New Time Frame
adjustedAcceleration = []
adjustedAccelerationCopy = accelData.copy()
for i in range(6, rangeMax - 1):
    adjustedAcceleration.append(adjustedAccelerationCopy[i])

#Adjusts Smoothed Acceleration Curve to Match New Time Frame
adjustedAccelerationData = []
adjustedAccelerationDataCopy = list(accelDataLine)
for i in range(6,rangeMax -1):
    adjustedAccelerationData.append(adjustedAccelerationDataCopy[i])

#Adjusts PP Curve to match New Time Frame
adjustedPPThresh = []
adjustedPPThreshCopy = list(currentDataFrame['pp_nr5'])
for i in range (6, rangeMax-1):
    adjustedPPThresh.append(adjustedPPThreshCopy[i])

print(adjustedTime[0])
print(adjustedAccelerationData[0])
x_animData = []
y_animData = []
#Ploting of Different Graphs: [ax1 = Acceleration Curve Data, ax2 = PP Data]
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time [s]', labelpad = 15, fontweight = 'bold', fontsize = 16)
ax1.set_ylabel('Acceleration Pressure [$^\circ$]', color = 'red', labelpad = 15, fontweight = 'bold', fontsize = 16)
ax1.plot(adjustedTime, adjustedAcceleration, color = 'red')
line, = ax1.plot(adjustedTime[0],adjustedAccelerationData[0], color = 'black', linestyle = 'dashed')

count = 0

ax1.axvspan(0,29, alpha=0.5, color = 'grey')

ax2 = ax1.twinx()
ax2.set_ylabel("Paranasal Persperation [$^\circ$$C^{2}$]", color = 'skyblue', labelpad=15, fontweight='bold', fontsize=16)
ppThresh = currentDataFrame['pp_nr5']
ax2.plot(adjustedTime, adjustedPPThresh, color = 'skyblue')
ax2.axhline(currentThreshold, linestyle = 'dotted', color = 'skyblue')

pointPP, = ax2.plot(adjustedTime[0], adjustedPPThresh[0], "o", color = 'lime')
x_animPoint = []
y_animPoint = []

def animation_frame(i):
    x_animData.append(int(adjustedTime[i]))
    y_animData.append(adjustedAccelerationData[i])

    line.set_xdata(x_animData)
    line.set_ydata(y_animData)

    x = adjustedTime[i]
    y = adjustedPPThresh[i]

    pointPP.set_xdata(x)
    pointPP.set_ydata(y)

    return line, pointPP,

fig.set_size_inches(20, 13, True)
dpi = 100

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,rangeMax-7), interval=100, repeat = False)

# def animation_frame2(j):
#     if(j > 106):
#         x_animData.append(int(adjustedTime[j]))
#         y_animData.append(adjustedAccelerationData[j])
#
#         line.set_xdata(x_animData)
#         line.set_ydata(y_animData)
#
#     return line,
#
# animation2 = FuncAnimation(fig, func=animation_frame2, frames = np.arange(0, rangeMax-100), interval=10, repeat = False)

# ax1.plot(adjustedTime, adjustedAccelerationData, color = 'black', linestyle = 'dashed')


# def animatePoint(i):
#     x = adjustedTime[i]
#     y = adjustedPPThresh[i]
#
#     pointPP.set_xdata(x)
#     pointPP.set_ydata(y)
#
#     return pointPP,

#pointAnim = FuncAnimation(fig, func = animatePoint, frames = np.arange(0, rangeMax-7), interval=250, repeat = False)

fig.legend(["Original Acceleration Line", "Adjusted Acceleration Line", "Start-up Zone", "Paranasal Perspiration (PP) Signal", "PP Threshold"])

#animation.save('smoothedCurve.mp4', writer=Writer, dpi=dpi)
plt.show()
