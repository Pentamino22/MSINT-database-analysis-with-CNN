import java.io.*;
import java.util.Random;

public class SimpleCNN {
    // Network dimensions
    private static final int INPUT_SIZE = 784;
    private static final int HIDDEN_SIZE_1 = 16;
    private static final int HIDDEN_SIZE_2 = 16;
    private static final int OUTPUT_SIZE = 10;
    // Hyperparameters
    private static final double LEARNING_RATE = 0.025;
    private static final int EPOCHS = 50;
    private static final int BATCH_SIZE = 50;
    
    // Activation function 1-Sigmoid 2-ReLU
    private static final int ACTIVATION_FUNCTION = 2;
    
    // Weights and biases
    private double[][] weights1 = new double[HIDDEN_SIZE_1][INPUT_SIZE];
    private double[][] weights2 = new double[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
    private double[][] weights3 = new double[OUTPUT_SIZE][HIDDEN_SIZE_2];
    private double[] bias1 = new double[HIDDEN_SIZE_1];
    private double[] bias2 = new double[HIDDEN_SIZE_2];
    private double[] bias3 = new double[OUTPUT_SIZE];
    

    
    private Random random = new Random();

    // Initialize weights and biases
    public SimpleCNN() {
        initializeWeights(weights1);
        initializeWeights(weights2);
        initializeWeights(weights3);
        initializeBias(bias1);
        initializeBias(bias2);
        initializeBias(bias3);
    }

    private void initializeWeights(double[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = random.nextGaussian() * 0.01; // Small random values
            }
        }
    }

    private void initializeBias(double[] bias) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0.0;
        }
    }

    // Forward propagation
    private double[] forward(double[] input) {
        double[] hidden1 = activate(matMul(weights1, input, bias1));
        double[] hidden2 = activate(matMul(weights2, hidden1, bias2));
        return softmax(matMul(weights3, hidden2, bias3));
    }

    // Matrix multiplication
    private double[] matMul(double[][] weights, double[] input, double[] bias) {
        double[] output = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i] += weights[i][j] * input[j];
            }
            output[i] += bias[i];
        }
        return output;
    }

    // Activation function 
    private double[] activate(double[] input) {
        double[] output = new double[input.length];
        if(ACTIVATION_FUNCTION == 2){
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }}
        else {
        	for (int i = 0; i < input.length; i++) {
             output[i] = 1.0 / (1.0 + Math.exp(-input[i])); 
        }
        }
        return output;
    }

    // Softmax function
    private double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;
        for (double value : input) {
            sum += Math.exp(value);
        }
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]) / sum;
        }
        return output;
    }

    // Loss function (Cross-entropy)
    private double computeLoss(double[] prediction, double[] label) {
        double loss = 0.0;
        for (int i = 0; i < prediction.length; i++) {
            loss -= label[i] * Math.log(prediction[i] + 1e-9);
        }
        return loss;
    }
    public void test(double[][] testData, double[][] testLabels) {
        int correct = 0;
        for (int i = 0; i < testData.length; i++) {
            double[] prediction = forward(testData[i]);
            int predictedLabel = getPredictedLabel(prediction);
            int actualLabel = getActualLabel(testLabels[i]);
            if (predictedLabel == actualLabel) {
                correct++;
            }
        }
        System.out.println();
     System.out.printf("Accuracy: %.2f percent",(correct / (double) testData.length) * 100);
       
    }

    private int getPredictedLabel(double[] output) {
        int label = 0;
        double max = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i];
                label = i;
            }
        }
        return label;
    }

    private int getActualLabel(double[] label) {
        for (int i = 0; i < label.length; i++) {
            if (label[i] == 1.0) {
                return i;
            }
        }
        return -1; // Error case
    }

    // Training function 
    public void trainMiniBatch(double[][] trainData, double[][] trainLabels,double[][] testData, double[][] testLabels) {
        int numSamples = trainData.length;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Shuffle the data
            shuffleData(trainData, trainLabels);

            double totalLoss = 0.0;
            int correctPredictions = 0;

            // Process data in mini-batches
            for (int batchStart = 0; batchStart < numSamples; batchStart += BATCH_SIZE) {
                int batchEnd = Math.min(batchStart + BATCH_SIZE, numSamples);
                int batchSize = batchEnd - batchStart;

                // Initialize accumulators for gradients
                double[][] gradW1 = new double[HIDDEN_SIZE_1][INPUT_SIZE];
                double[][] gradW2 = new double[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
                double[][] gradW3 = new double[OUTPUT_SIZE][HIDDEN_SIZE_2];
                double[] gradB1 = new double[HIDDEN_SIZE_1];
                double[] gradB2 = new double[HIDDEN_SIZE_2];
                double[] gradB3 = new double[OUTPUT_SIZE];

                // Accumulate gradients for the mini-batch
                for (int i = batchStart; i < batchEnd; i++) {
                    double[] input = trainData[i];
                    double[] label = trainLabels[i];

                    // Forward pass
                    double[] hidden1 = activate(matMul(weights1, input, bias1));
                    double[] hidden2 = activate(matMul(weights2, hidden1, bias2));
                    double[] output = softmax(matMul(weights3, hidden2, bias3));

                    // Calculate loss for monitoring
                    totalLoss += computeLoss(output, label);

                    // Check if the prediction is correct
                    if (getPredictedLabel(output) == getActualLabel(label)) {
                        correctPredictions++;
                    }

                    // Backpropagation steps as before
                    double[] outputError = new double[OUTPUT_SIZE];
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        outputError[j] = output[j] - label[j];
                    }

                    double[] hidden2Error = new double[HIDDEN_SIZE_2];
                    for (int j = 0; j < HIDDEN_SIZE_2; j++) {
                        for (int k = 0; k < OUTPUT_SIZE; k++) {
                            hidden2Error[j] += weights3[k][j] * outputError[k];
                        }
                      
                        if(ACTIVATION_FUNCTION == 2){ hidden2Error[j] *= (hidden2[j] > 0 ? 1 : 0);} // ReLu derivative
                        else { hidden2Error[j] *= hidden2[j] * (1 - hidden2[j]);} //Sigmoid derivative
                    }

                    double[] hidden1Error = new double[HIDDEN_SIZE_1];
                    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
                        for (int k = 0; k < HIDDEN_SIZE_2; k++) {
                            hidden1Error[j] += weights2[k][j] * hidden2Error[k];
                        }
                        
                        if(ACTIVATION_FUNCTION == 2){  hidden1Error[j] *= (hidden1[j] > 0 ? 1 : 0);} // ReLu derivative
                        else { hidden1Error[j] *= hidden1[j] * (1 - hidden1[j]); } //Sigmoid derivative
                    }

                    // Accumulate gradients (Output Layer)
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        for (int k = 0; k < HIDDEN_SIZE_2; k++) {
                            gradW3[j][k] += outputError[j] * hidden2[k];
                        }
                        gradB3[j] += outputError[j];
                    }

                    // Accumulate gradients (Hidden Layer 2)
                    for (int j = 0; j < HIDDEN_SIZE_2; j++) {
                        for (int k = 0; k < HIDDEN_SIZE_1; k++) {
                            gradW2[j][k] += hidden2Error[j] * hidden1[k];
                        }
                        gradB2[j] += hidden2Error[j];
                    }

                    // Accumulate gradients (Hidden Layer 1)
                    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
                        for (int k = 0; k < INPUT_SIZE; k++) {
                            gradW1[j][k] += hidden1Error[j] * input[k];
                        }
                        gradB1[j] += hidden1Error[j];
                    }
                }

                // Update weights and biases using SGD
                for (int i = 0; i < HIDDEN_SIZE_1; i++) {
                    for (int j = 0; j < INPUT_SIZE; j++) {
                        weights1[i][j] -= LEARNING_RATE * gradW1[i][j] / batchSize;
                    }
                    bias1[i] -= LEARNING_RATE * gradB1[i] / batchSize;
                }

                for (int i = 0; i < HIDDEN_SIZE_2; i++) {
                    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
                        weights2[i][j] -= LEARNING_RATE * gradW2[i][j] / batchSize;
                    }
                    bias2[i] -= LEARNING_RATE * gradB2[i] / batchSize;
                }

                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    for (int j = 0; j < HIDDEN_SIZE_2; j++) {
                        weights3[i][j] -= LEARNING_RATE * gradW3[i][j] / batchSize;
                    }
                    bias3[i] -= LEARNING_RATE * gradB3[i] / batchSize;
                }
            }
            int correct = 0;
            for (int i = 0; i < testData.length; i++) {
                double[] prediction = forward(testData[i]);
                int predictedLabel = getPredictedLabel(prediction);
                int actualLabel = getActualLabel(testLabels[i]);
                if (predictedLabel == actualLabel) {
                    correct++;
                }
            }
            System.out.println();
            System.out.printf("Epoch "+(epoch+1)+" Accuracy = %.2f percent",(correct / (double) testData.length) * 100);
       
           // double accuracy = (correctPredictions / (double) numSamples) * 100.0;
          //  System.out.printf("{"+(epoch+1)+";%.2f};",accuracy);
            //  System.out.println("Epoch " + (epoch + 1) + ": Loss = " + totalLoss / numSamples + ", Accuracy = " + accuracy + "%");
        }
    }


    // Shuffle the data and labels in unison
    private void shuffleData(double[][] data, double[][] labels) {
        Random rand = new Random();
        for (int i = data.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            // Swap data
            double[] tempData = data[i];
            data[i] = data[j];
            data[j] = tempData;

            // Swap labels
            double[] tempLabel = labels[i];
            labels[i] = labels[j];
            labels[j] = tempLabel;
        }
    }

    public void saveModel(String fileName) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Save weights1
            writer.write("weights1\n");
            for (double[] row : weights1) {
                for (double value : row) {
                    writer.write(value + " ");
                }
                writer.newLine();
            }

            // Save bias1
            writer.write("bias1\n");
            for (double value : bias1) {
                writer.write(value + " ");
            }
            writer.newLine();

            // Save weights2
            writer.write("weights2\n");
            for (double[] row : weights2) {
                for (double value : row) {
                    writer.write(value + " ");
                }
                writer.newLine();
            }

            // Save bias2
            writer.write("bias2\n");
            for (double value : bias2) {
                writer.write(value + " ");
            }
            writer.newLine();

            // Save weights3
            writer.write("weights3\n");
            for (double[] row : weights3) {
                for (double value : row) {
                    writer.write(value + " ");
                }
                writer.newLine();
            }

            // Save bias3
            writer.write("bias3\n");
            for (double value : bias3) {
                writer.write(value + " ");
            }
            writer.newLine();

            System.out.println("Model saved to " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        SimpleCNN cnn = new SimpleCNN();
        MNISTLoader loader = new MNISTLoader();
        System.out.println("Initialasing data. Please Wait.");
        // Load training and test data
        double[][] trainData = loader.loadImages("train-images.idx3-ubyte", 60000);
        double[][] trainLabels = loader.loadLabels("train-labels.idx1-ubyte", 60000);
        double[][] testData = loader.loadImages("t10k-images.idx3-ubyte", 10000);
        double[][] testLabels = loader.loadLabels("t10k-labels.idx1-ubyte", 10000);
        System.out.println("Training process begins");
        // Train and test the model
        cnn.trainMiniBatch(trainData, trainLabels,testData,testLabels);
        cnn.test(testData, testLabels);
        System.out.println();
        cnn.saveModel("data.txt");
    }
}
