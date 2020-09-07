class testDataPreprocessing:

    def checkColumns(self, i, testArray):
        column = i
        for j in range(28):
            if testArray[j][i] != 0:
                column = 999
        return column

    def createTestData(self, start, end, testArray):
        arr = [[1 for j in range(28)] for i in range(28)]
        for i in range(28):
            for j in range(28):
                arr[i][j] = 0
        for i in range(28):
            mid = 14-int(((end-start)/2))
            # for j in range(x, (x+(z-y))):
            for j in range(start, end+1):
                arr[i][mid] = testArray[i][j]
                mid += 1
        return arr

    def findEmptyColumns(self, columns):
        emptyColumns = []
        for i in range(28):
            if columns[i] != 999:
                emptyColumns.append(i)
        return emptyColumns

    def separatingNumbersAndPredicting(self, emptyColumns, testArray, CNN_model):
        predictedNumber = ""
        start = 0
        end = 27
        for i in range(len(emptyColumns)):
            if start == emptyColumns[i]:
                pass
            elif start < emptyColumns[i]:
                if start + 1 == emptyColumns[i]:
                    start = emptyColumns[i]
                else:
                    # Predicting the number
                    predictedNumber += str(CNN_model.predictNumber(self.createTestData(start, emptyColumns[i], testArray)))
                    start = emptyColumns[i]
        if (start == end):
            pass
        else:
            # Predicting the number
            predictedNumber += str(CNN_model.predictNumber(self.createTestData(start, 27, testArray)))
        return predictedNumber


if __name__ == "__main__":
    print("Used to preprocess the Mixed Numbers data")
else:
    print("testDataPreprocessing is successfully imported")