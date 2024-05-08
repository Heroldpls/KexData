import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.colors as mcolors


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

    print("Averages for Young: " + str(avgYoung/15), ", Middle " + str(avgMiddle/15), ", Old: "+ str(avgOld/15))  # /15 for accuracy
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

    print(nrOld, nrMiddle, nrYoung)

    resultsOld = [x / nrOld * 100 for x in resultsOld]
    resultsMiddle = [x / nrMiddle * 100 for x in resultsMiddle]
    resultsYoung = [x / nrYoung * 100 for x in resultsYoung]

    print(resultsYoung)
    print(resultsMiddle)
    print(resultsOld)

    print("-----")

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
accuracyPerImgAndAge()

def accuracyPerImgAndAgeOnlySomeImg():

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

    print(nrOld, nrMiddle, nrYoung)

    resultsOld = [x / nrOld * 100 for x in resultsOld]
    resultsMiddle = [x / nrMiddle * 100 for x in resultsMiddle]
    resultsYoung = [x / nrYoung * 100 for x in resultsYoung]



    print("-----")
    resultsYoung = [resultsYoung[0], resultsYoung[2], resultsYoung[4]]
    resultsMiddle = [resultsMiddle[0], resultsMiddle[2], resultsMiddle[4]]
    resultsOld = [resultsOld[0], resultsOld[2], resultsOld[4]]

    print(resultsYoung)
    print(resultsMiddle)
    print(resultsOld)

    # Combine accuracy values into one list for each group
    combined_accuracy_values = [[resultsYoung[i], resultsMiddle[i], resultsOld[i]] for i in
                                range(len(resultsYoung))]

    # Generate x values for 15 groups
    x = np.arange(1, 6, 2)

    # Plot the bars
    plt.bar(x - 0.4, [values[0] for values in combined_accuracy_values], width=0.4, color='skyblue', label=str(YOUNG_INTERVAL_LOWER_LIMIT) + " - " + str(MIDDLE_INTERVAL_LOWER_LIMIT- 1) + " year olds")
    plt.bar(x, [values[1] for values in combined_accuracy_values], width=0.4, color='salmon', label=str(MIDDLE_INTERVAL_LOWER_LIMIT) + " - " + str(OLD_INTERVAL_LOWER_LIMIT- 1) + " year olds")
    plt.bar(x + 0.4, [values[2] for values in combined_accuracy_values], width=0.4, color='lightgreen', label=str(OLD_INTERVAL_LOWER_LIMIT) + "+  year olds")

    # Add labels and title
    plt.xlabel('Image (nr)')
    plt.ylabel('Accuracy (%)')
    plt.title('Standout age-statistics')
    plt.xticks(x, x)
    plt.yticks(range(0, 105, 10))
    plt.legend()

    # Show plot
    plt.show()
#accuracyPerImgAndAgeOnlySomeImg()

def accuracyPerImg():
    resultsAll = [0] * 15
    totalParticipants = 0

    for result in results[1:]:
        imgIndex = 0
        for i in range(15, 103, 6):
            if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT:
                resultsAll[imgIndex] += float(result[i].split("/")[0])
                imgIndex += 1
                totalParticipants += 1

    totalParticipants /= 15

    resultsAll = [x / totalParticipants * 100 for x in resultsAll]
    generalAcc = 0
    for result in resultsAll:
        generalAcc += result
    generalAcc /= 15

    print(resultsAll)
    print("General accuracy: " + str(generalAcc))
    # Combine accuracy values into one list for each group
    combined_accuracy_values = [[resultsAll[i]] for i in range(len(resultsAll))]


    real_images = [1,3,4,6,7,9,10,12,13,15]
    real_accuracy_values = [combined_accuracy_values[i - 1] for i in real_images]

    fake_images = [2,5,8,11,14]
    fake_accuracy_values = [combined_accuracy_values[i - 1] for i in fake_images]
    # Generate x values for 15 groups
    x = np.arange(1, 16)

    # Plot the bars
    light_green = mcolors.to_rgb('lightgreen')
    darker_green = tuple(max(0, c - 0.1) for c in light_green) #make slightly darker green.
    colors = [darker_green] * 15  # Set default color for all bars
    # Change color for specific bars (for example, bars 1, 5, and 10)
    #specific_bars = [2, 5, 8, 11, 14]
    """
    for bar_index in specific_bars:
        colors[bar_index - 1] = 'salmon'  # Adjust index to match Python's 0-based indexing
    """


    plt.bar(real_images, [values[0] for values in real_accuracy_values], width=0.5, color=darker_green,
            label="REAL")
    plt.bar(fake_images, [values[0] for values in fake_accuracy_values], width=0.5, color="salmon",
            label="AI")

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
#print("Image specific accuracy" + str(imgAccuracy(2)))

def accuracyPerAgeRegression():

    x_Old = []
    x_Middle = []
    x_Young = []
    y_Old = []
    y_Middle = []
    y_Young = []

    nrOld = 0
    nrMiddle = 0
    nrYoung = 0


    for result in results[1:]:
        score = 0
        for i in range(15, 103, 6):
            score += float(result[i].split("/")[0])
        accuracy = score / 15
        if int(result[2]) >= OLD_INTERVAL_LOWER_LIMIT:
            nrOld += 1
            x_Old.append(int(result[2]))
            y_Old.append(accuracy)
        elif int(result[2]) >= MIDDLE_INTERVAL_LOWER_LIMIT:
            nrMiddle += 1
            x_Middle.append(int(result[2]))
            y_Middle.append(accuracy)

        elif int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT:
            nrYoung += 1
            x_Young.append(int(result[2]))
            y_Young.append(accuracy)



    print(nrOld, nrMiddle, nrYoung)

    # Convert to %
    y_Old = [a * 100 for a in y_Old]
    y_Middle = [a * 100 for a in y_Middle]
    y_Young = [a * 100 for a in y_Young]

    print(y_Young)
    print(y_Middle)
    print(y_Old)

    #For the regression line
    x = np.array(x_Young + x_Middle + x_Old)
    y = np.array(y_Young + y_Middle + y_Old)
    coefficients = np.polyfit(x, y, 1)  # Performing linear regression (1st degree polynomial)
    line = np.polyval(coefficients, x)  # Generating y-values for the line

    # Plotting the regression line
    plt.plot(x, line, color='black', label='Linear Regression')
    # Creating scatter plot
    plt.scatter(x_Young, y_Young, color='skyblue', alpha=0.8,  label=str(YOUNG_INTERVAL_LOWER_LIMIT) + " - " + str(MIDDLE_INTERVAL_LOWER_LIMIT- 1) + " year olds")  # alpha controls transparency
    plt.scatter(x_Middle, y_Middle, color='salmon', alpha=0.8, label='Data')  # alpha controls transparency
    plt.scatter(x_Old, y_Old, color='lightgreen', alpha=0.8, label='Data')  # alpha controls transparency

    labels = [
        "Regression line",
        str(YOUNG_INTERVAL_LOWER_LIMIT) + " - " + str(MIDDLE_INTERVAL_LOWER_LIMIT - 1) + " year olds",
        str(MIDDLE_INTERVAL_LOWER_LIMIT) + " - " + str(OLD_INTERVAL_LOWER_LIMIT - 1) + " year olds",
        str(OLD_INTERVAL_LOWER_LIMIT) + "+ year olds"
    ]

    plt.legend(labels=labels, loc='upper right')

    # Adding labels and title
    plt.title('Linear Regression for accuracy in regards to age')
    plt.xlabel('Age (years)')
    plt.ylabel('Accuracy (%)')

    plt.show()
accuracyPerAgeRegression()

def confidenceToAccuracyRegression():
    x = []
    y = []

    participants = 0
    for result in results[1:]:
        if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT: # In oder to remove some unwanted data (from people aged <18):
            score = 0
            for i in range(15, 103, 6):
                score += float(result[i].split("/")[0])

            x.append(int(result[5])) #Add the confidence number to x.
            accuracy = score / 15
            y.append(accuracy)
            participants += 1

    y = [a * 100 for a in y]

    coefficients = np.polyfit(x, y, 1)  # Performing linear regression (1st degree polynomial)
    line = np.polyval(coefficients, x)  # Generating y-values for the line

    # Plotting the regression line
    plt.plot(x, line, color='black', label='Linear Regression')
    # Creating scatter plot
    plt.scatter(x, y, color='skyblue', alpha=0.8, label='Data')  # alpha controls transparency


    # Adding labels and title
    plt.title('Linear Regression for accuracy in regards to perceived confidence')
    plt.xlabel('Confidence (1-6)')
    plt.ylabel('Accuracy (%)')

    plt.show()
#confidenceToAccuracyRegression()

def confidenceAverageAccuracy():
    avgAccuracy = [0] * 6

    participants_1 = 0
    participants_2 = 0
    participants_3 = 0
    participants_4 = 0
    participants_5 = 0
    participants_6 = 0

    for result in results[1:]:
        if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT:  # In oder to remove some unwanted data (from people aged <18):
            match int(result[5]):
                case 1:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[0] += accuracy
                    participants_1 += 1


                case 2:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[1] += accuracy
                    participants_2 += 1


                case 3:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[2] += accuracy
                    participants_3 += 1


                case 4:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[3] += accuracy
                    participants_4 += 1


                case 5:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[4] += accuracy
                    participants_5 += 1


                case 6:
                    score = 0
                    for i in range(15, 103, 6):
                        score += float(result[i].split("/")[0])
                    accuracy = score / 15
                    avgAccuracy[5] += accuracy
                    participants_6 += 1


    avgAccuracy[0] /= participants_1
    avgAccuracy[1] /= participants_2
    avgAccuracy[2] /= participants_3
    avgAccuracy[3] /= participants_4
    avgAccuracy[4] /= participants_5
    avgAccuracy[5] /= participants_6

    avgAccuracy = [x * 100 for x in avgAccuracy]
    print(avgAccuracy)
#confidenceAverageAccuracy()

def aiToRealGuessRatio():

    totalGuesses = 0
    totalAIGuesses = 0

    for result in results[1:]:
        if int(result[2]) >= YOUNG_INTERVAL_LOWER_LIMIT:  # In order to remove some unwanted data (from people aged <18):
            score = 0
            for i in range(14, 103, 6):
                totalGuesses += 1
                if result[i] == "Ja":
                    totalAIGuesses += 1

    print(totalAIGuesses/totalGuesses)
#aiToRealGuessRatio()