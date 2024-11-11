package main

import (
	"blueprint"
	"fmt"
	"log"
	"os"
)

// Declare bp as a global variable
var bp *blueprint.Blueprint
var Images [][]byte
var Labels []byte

const baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

func mnistStart() {
	modelMnistSetup()
	mnistSetup()
}

func mnistSetup() {
	// Create the directory for MNIST images
	imgDir := "./host/MNIST/images"
	dataFile := "./host/MNIST/mnist_data.json"
	imgWidth, imgHeight := 28, 28 // Dimensions for MNIST images

	if err := os.MkdirAll(imgDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create MNIST image directory: %v", err)
	}

	// Ensure MNIST data is downloaded and unzipped
	if err := EnsureMNISTDownloads(); err != nil {
		log.Fatalf("Failed to ensure MNIST downloads: %v", err)
	}

	// Load the MNIST images and labels
	LoadMNIST()

	// Convert labels to integer slice for compatibility
	intLabels := bp.ConvertLabelsToInts(Labels)

	// Save images and labels to JPEGs and JSON data file
	if err := bp.SaveImagesAndData(Images, intLabels, imgDir, dataFile, imgWidth, imgHeight); err != nil {
		log.Fatalf("Failed to save MNIST images and data: %v", err)
	}

	fmt.Println("MNIST setup completed: images and labels saved.")
}

// EnsureMNISTDownloads ensures that the MNIST dataset is downloaded and unzipped correctly
func EnsureMNISTDownloads() error {
	// Updated file links from Google's storage
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}

	for _, file := range files {
		localFile := file
		if _, err := os.Stat(localFile); os.IsNotExist(err) {
			log.Printf("Downloading %s...\n", file)
			if err := bp.DownloadFile(localFile, baseURL+file); err != nil {
				return err
			}
			log.Printf("Downloaded %s\n", file)

			// Unzip the file
			if err := bp.UnzipFile(localFile); err != nil {
				return err
			}
		} else {
			log.Printf("%s already exists, skipping download.\n", file)
		}
	}
	return nil
}

func LoadMNIST() {
	var err error
	Images, err = bp.LoadBinaryDatasetImages("train-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("failed to load training images: %v", err)
	}

	Labels, err = bp.LoadLabels("train-labels-idx1-ubyte")
	if err != nil {
		fmt.Errorf("failed to load training labels: %w", err)
	}
}

// modelSetup initializes the Blueprint instance and sets up the model configuration.
func modelMnistSetup() {
	// Initialize bp with a new Blueprint instance
	bp = blueprint.NewBlueprint(nil)

	// Configure model parameters
	numInputs := 28 * 28        // Example for MNIST data, 28x28 images
	numHiddenNeurons := 28 * 28 // Number of neurons in the hidden layer
	numOutputs := 10            // Number of classes (0-9 for MNIST)
	outputActivationTypes := []string{
		"sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid",
		"sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid",
	} // Activation type for output layer

	modelID := "mnist-model-001"
	projectName := "MNIST Digit Classification"

	// Call CreateCustomNetworkConfig to set up the model structure
	bp.CreateCustomNetworkConfig(numInputs, numHiddenNeurons, numOutputs, outputActivationTypes, modelID, projectName)
	fmt.Println("Model setup completed.")
	fmt.Printf("Total Neurons: %d, Total Layers: %d\n", bp.Config.Metadata.TotalNeurons, bp.Config.Metadata.TotalLayers)
}
