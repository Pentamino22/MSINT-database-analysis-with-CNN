import java.io.*;

public class MNISTLoader {
    public static double[][] loadImages(String filePath, int numImages) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int magicNumber = dis.readInt();
            int numberOfImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            int imageSize = rows * cols;
            double[][] images = new double[numImages][imageSize];

            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < imageSize; j++) {
                    images[i][j] = dis.readUnsignedByte() / 255.0;
                }
            }
            return images;
        }
    }

    public static double[][] loadLabels(String filePath, int numLabels) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int magicNumber = dis.readInt();
            int numberOfLabels = dis.readInt();

            double[][] labels = new double[numLabels][10];
            for (int i = 0; i < numLabels; i++) {
                int label = dis.readUnsignedByte();
                labels[i][label] = 1.0; 
            }
            return labels;
        }
    }
}
