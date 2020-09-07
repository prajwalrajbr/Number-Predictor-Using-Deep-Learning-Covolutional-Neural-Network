from numberPredictorCNN import numberPredictorCNN
from drawNumber import drawNumber, testArray
from testDataPreprocessing import testDataPreprocessing

if __name__ == "__main__":

    draw = drawNumber()
    CNN_model = numberPredictorCNN()
    testDP = testDataPreprocessing()

    # Train the model
    CNN_model.trainModel()

    # Show Board to draw Number
    draw.showBoard()

    # Store the column values
    columns = []
    for j in range(56):
        columns.append(testDP.checkColumns(j, testArray))

    # Store the empty columns
    emptyColumns = testDP.findEmptyColumns(columns)

    # Predicting the separated numbers
    predictedNumber = testDP.separatingNumbersAndPredicting(emptyColumns, testArray, CNN_model)

    # Output the predicted Number
    print("Predicted Number = " + predictedNumber)
    draw.showPredictedNo(predictedNumber)
