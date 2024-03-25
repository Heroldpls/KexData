import matplotlib.pyplot as plt
import numpy as np
import csv

file_path = "Results.csv"

results = []
OLD_INTERVAL_LOWER_LIMIT = 60
MIDDLE_INTERVAL_LOWER_LIMIT = 40
YOUNG_INTERVAL_LOWER_LIMIT = 18

def readData():
    rowIndex = 0
    with open(file_path, mode='r', newline='', encoding = 'utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:

            if (rowIndex == 0):

                index = 0
                """
                # For checking column names and numbers
                for data in row:
    
                    print(str(index) + ". " + data)
                    index += 1
                
                print(row[0])
                """
            results.append(row)
            rowIndex += 1
readData()


#  Calculates the average score of each age interval.
def calcAvgScoreAgeIntervals():
    resultsOld = []
    resultsMiddle = []
    resultsYoung = []

    for result in results[1:]:

        if int(result[2]) >= OLD_INTERVAL_LOWER_LIMIT:
            resultsOld.append(float(result[1].split("/")[0]))
        elif int(result[2]) >= MIDDLE_INTERVAL_LOWER_LIMIT:
            resultsMiddle.append(float(result[1].split("/")[0]))
        elif int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT :
            resultsYoung.append(float(result[1].split("/")[0]))

    avgOld = 0
    for result in resultsOld:
        avgOld += result
    avgOld /= len(resultsOld)

    avgMiddle = 0
    for result in resultsMiddle:
        avgMiddle += result
    avgMiddle /= len(resultsMiddle
                     )
    avgYoung = 0
    for result in resultsYoung:
        avgYoung += result
    avgYoung /= len(resultsYoung)

    print("Young: " + str(avgYoung/15), ", Middle " + str(avgMiddle/15), ", Old: "+ str(avgOld/15))  # /15 for accuracy
#calcAvgScoreAgeIntervals()

def accuracyPerImgAndAge():

    resultsOld = [0] * 15
    resultsMiddle = [0] * 15
    resultsYoung = [0] * 15

    nrOld = 0
    nrMiddle = 0
    nrYoung = 0

    for result in results[1:]:
        imgIndex = 0
        for i in range(15, 103, 6):
            if int(result[2]) >= OLD_INTERVAL_LOWER_LIMIT:
                nrOld += 1
                resultsOld[imgIndex] += float(result[i].split("/")[0])
            elif int(result[2]) >= MIDDLE_INTERVAL_LOWER_LIMIT:
                nrMiddle += 1
                resultsMiddle[imgIndex] += float(result[i].split("/")[0])
            elif int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT :
                nrYoung += 1
                resultsYoung[imgIndex] += float(result[i].split("/")[0])

            imgIndex += 1

    nrOld /= 15
    nrMiddle /= 15
    nrYoung /= 15

    resultsOld = [x / nrOld * 100 for x in resultsOld]
    resultsMiddle = [x / nrMiddle * 100 for x in resultsMiddle]
    resultsYoung = [x / nrYoung * 100 for x in resultsYoung]

    # Combine accuracy values into one list for each group
    combined_accuracy_values = [[resultsYoung[i], resultsMiddle[i], resultsOld[i]] for i in
                                range(len(resultsYoung))]

    # Generate x values for 15 groups
    x = np.arange(1, 16)

    # Plot the bars
    plt.bar(x - 0.2, [values[0] for values in combined_accuracy_values], width=0.2, color='skyblue', label=str(YOUNG_INTERVAL_LOWER_LIMIT) + " - " + str(MIDDLE_INTERVAL_LOWER_LIMIT- 1) + " year olds")
    plt.bar(x, [values[1] for values in combined_accuracy_values], width=0.2, color='salmon', label=str(MIDDLE_INTERVAL_LOWER_LIMIT) + " - " + str(OLD_INTERVAL_LOWER_LIMIT- 1) + " year olds")
    plt.bar(x + 0.2, [values[2] for values in combined_accuracy_values], width=0.2, color='lightgreen', label=str(OLD_INTERVAL_LOWER_LIMIT) + "+  year olds")

    # Add labels and title
    plt.xlabel('Image (nr)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracies grouped by age and image')
    plt.xticks(x, x)
    plt.yticks(range(0, 105, 10))
    plt.legend()

    # Show plot
    plt.show()
#accuracyPerImgAndAge()

def accuracyPerImg():
    resultsAll = [0] * 15
    totalParticipants = 0

    for result in results[1:]:
        imgIndex = 0
        for i in range(15, 103, 6):
            if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT :
                resultsAll[imgIndex] += float(result[i].split("/")[0])
                imgIndex += 1
                totalParticipants += 1

    totalParticipants /= 15

    resultsAll = [x / totalParticipants * 100 for x in resultsAll]

    # Combine accuracy values into one list for each group
    combined_accuracy_values = [[resultsAll[i]] for i in
                                range(len(resultsAll))]

    # Generate x values for 15 groups
    x = np.arange(1, 16)

    # Plot the bars
    plt.bar(x - 0.2, [values[0] for values in combined_accuracy_values], width=0.2, color='skyblue',
            label=str(YOUNG_INTERVAL_LOWER_LIMIT) + "+ year olds")

    # Add labels and title
    plt.xlabel('Image (nr)')
    plt.ylabel('Accuracy (%)')
    plt.title('Specific image accuracies without age distinction')
    plt.xticks(x, x)
    plt.yticks(range(0, 105, 10))
    plt.legend()

    # Show plot
    plt.show()
#accuracyPerImg()

def imgAccuracy(imageNr): # Hardest image: nr 15, Easiest image: nr 2.
    imgScore = 0
    totalParticipants = 0
    for result in results[1:]:
        if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT:
            imgScore += float(result[15 + (imageNr - 1) * 6].split("/")[0])
            totalParticipants += 1

    imgAccuracy = imgScore / totalParticipants  * 100
    return imgAccuracy
#print(imgAccuracy(15))
