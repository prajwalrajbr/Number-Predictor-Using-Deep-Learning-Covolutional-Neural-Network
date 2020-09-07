from drawNumber import drawNumber, testArray
from numberPredictorCNN import numberPredictorCNN

if __name__ == "__main__":
    draw = drawNumber()
    CNN_model = numberPredictorCNN()

    CNN_model.trainModel()
    draw.showBoard()

    predictedNumber = str(CNN_model.predictNumber(testArray))

    draw.showPredictedNo(predictedNumber)
