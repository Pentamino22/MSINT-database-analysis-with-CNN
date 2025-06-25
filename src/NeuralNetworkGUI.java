import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class NeuralNetworkGUI extends JFrame {
	
    // Network dimensions
    private static final int INPUT_SIZE = 784;
    private static final int HIDDEN_SIZE_1 = 16;
    private static final int HIDDEN_SIZE_2 = 16;
    private static final int OUTPUT_SIZE = 10;
    private static final int IMAGE_SIZE = 28;

    // Weights and biases
    private double[][] weights1 = new double[HIDDEN_SIZE_1][INPUT_SIZE];
    private double[][] weights2 = new double[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
    private double[][] weights3 = new double[OUTPUT_SIZE][HIDDEN_SIZE_2];
    private double[] bias1 = new double[HIDDEN_SIZE_1];
    private double[] bias2 = new double[HIDDEN_SIZE_2];
    private double[] bias3 = new double[OUTPUT_SIZE];
    private JLabel imageLabel;
    private JLabel predictionLabel;

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
    public void loadModel(String fileName) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            String section = "";

            while ((line = reader.readLine()) != null) {
                if (line.equals("weights1")) {
                    section = "weights1";
                    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
                        String[] values = reader.readLine().split(" ");
                        for (int j = 0; j < INPUT_SIZE; j++) {
                            weights1[i][j] = Double.parseDouble(values[j]);
                        }
                    }
                } else if (line.equals("bias1")) {
                    section = "bias1";
                    String[] values = reader.readLine().split(" ");
                    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
                        bias1[i] = Double.parseDouble(values[i]);
                    }
                } else if (line.equals("weights2")) {
                    section = "weights2";
                    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
                        String[] values = reader.readLine().split(" ");
                        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
                            weights2[i][j] = Double.parseDouble(values[j]);
                        }
                    }
                } else if (line.equals("bias2")) {
                    section = "bias2";
                    String[] values = reader.readLine().split(" ");
                    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
                        bias2[i] = Double.parseDouble(values[i]);
                    }
                } else if (line.equals("weights3")) {
                    section = "weights3";
                    for (int i = 0; i < OUTPUT_SIZE; i++) {
                        String[] values = reader.readLine().split(" ");
                        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
                            weights3[i][j] = Double.parseDouble(values[j]);
                        }
                    }
                } else if (line.equals("bias3")) {
                    section = "bias3";
                    String[] values = reader.readLine().split(" ");
                    for (int i = 0; i < OUTPUT_SIZE; i++) {
                        bias3[i] = Double.parseDouble(values[i]);
                    }
                }
            }
            System.out.println("Model loaded from " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }  
    public NeuralNetworkGUI(double[][] testData,double[][] testLabels) {
        setTitle("Neural Network MNIST Predictor");
        setSize(400, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        imageLabel = new JLabel();
        imageLabel.setHorizontalAlignment(SwingConstants.CENTER);
        add(imageLabel, BorderLayout.CENTER);

        predictionLabel = new JLabel("Prediction: ");
        predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        add(predictionLabel, BorderLayout.SOUTH);

        JButton testButton = new JButton("Test Random Image");
        testButton.addActionListener(e -> testRandomImage(testData,testLabels));
        add(testButton, BorderLayout.NORTH);
    }
    // Activation function (ReLU)
    private double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
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
 // Get the predicted label (index of the highest probability)
    private int getPredictedLabel(double[] output) {
        int predictedLabel = 0;
        double maxProbability = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxProbability) {
                maxProbability = output[i];
                predictedLabel = i;
            }
        }
        return predictedLabel;
    }

    private int getActualLabel(double[] label) {
        for (int i = 0; i < label.length; i++) {
            if (label[i] == 1.0) {
                return i;
            }
        }
        throw new IllegalArgumentException("Invalid label: no 1 found in one-hot encoding.");
    }
    private void testRandomImage(double[][] testData,double[][] testLabels) {
        // Get a random image from the test set
    	 
        Random random = new Random();
        int randomIndex = (int) (Math.random() * 10000);
        double[] image = testData[randomIndex];
        double[] label = testLabels[randomIndex];
        // Predict using the model
        double[] hidden1 = activate(matMul(weights1, image, bias1));
        double[] hidden2 = activate(matMul(weights2, hidden1, bias2));
        double[] output = softmax(matMul(weights3, hidden2, bias3));

        int predictedLabel = getPredictedLabel(output);
        int actualLabel = getActualLabel(label);

        // Display the image
        BufferedImage img = renderImage(image);
        imageLabel.setIcon(new ImageIcon(img));

        // Display the prediction
        predictionLabel.setText("Prediction: " + predictedLabel + " (Actual: " + actualLabel + ")");
    }

//    private BufferedImage renderImage(double[] image) {
//        BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
//        for (int i = 0; i < 28; i++) {
//            for (int j = 0; j < 28; j++) {
//                int gray = (int) (image[i * 28 + j] * 255);
//                int rgb = new Color(gray, gray, gray).getRGB();
//                img.setRGB(j, i, rgb);
//            }
//        }
//        return img;
//    }
    
 // Render a flattened image (28x28) into a scaled BufferedImage
    private BufferedImage renderImage(double[] image) {
        int scaleFactor = 10; // Scale factor (e.g., 10 for 280x280 pixels)
        int scaledSize = IMAGE_SIZE * scaleFactor;

        BufferedImage scaledImage = new BufferedImage(scaledSize, scaledSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = scaledImage.createGraphics();

        // Create the original 28x28 image
        BufferedImage originalImage = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int pixelIndex = y * IMAGE_SIZE + x;
                int grayValue = (int) (image[pixelIndex] * 255); // Scale [0, 1] to [0, 255]
                int rgb = new Color(grayValue, grayValue, grayValue).getRGB();
                originalImage.setRGB(x, y, rgb);
            }
        }

        // Draw the original image onto the scaled image
        g2d.drawImage(originalImage, 0, 0, scaledSize, scaledSize, null);
        g2d.dispose();

        return scaledImage;
    }

    

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            
            MNISTLoader loader = new MNISTLoader();
        	double[][] testData = null;
    		try {
    			testData = loader.loadImages("t10k-images.idx3-ubyte", 10000);
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
            double[][] testLabels = null;
    		try {
    			testLabels = loader.loadLabels("t10k-labels.idx1-ubyte", 10000);
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
    		NeuralNetworkGUI gui = new NeuralNetworkGUI(testData,testLabels);
            gui.loadModel("data.txt");
            gui.setVisible(true);
        });
    }
}
