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

// Declare slices to store training and testing sessions
var TrainingSessions []blueprint.TrainingSession
var TestingSessions []blueprint.TrainingSession

const baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

func mnistStart() {
	modelMnistSetup()
	mnistSetup()
	setupModelTrainingSession()
	evaluateModelPerformance()
}

func mnistSetup() {
	// Create the directory for MNIST images
	imgDir := "./host/MNIST/images"
	dataFile := "./host/MNIST/mnist_data.json"
	imgWidth, imgHeight := 28, 28 // Dimensions for MNIST images

	// Check if the data file already exists
	if _, err := os.Stat(dataFile); err == nil {
		fmt.Println("MNIST data file already exists. Skipping image generation and JSON creation.")
		LoadMNIST()
		return
	}

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

	// Set the forgiveness threshold
	bp.Config.Metadata.ForgivenessThreshold = 0.8 // Example: 80% tolerance threshold

	// Call CreateCustomNetworkConfig to set up the model structure
	bp.CreateCustomNetworkConfig(numInputs, numHiddenNeurons, numOutputs, outputActivationTypes, modelID, projectName)
	fmt.Println("Model setup completed.")
	fmt.Printf("Total Neurons: %d, Total Layers: %d\n", bp.Config.Metadata.TotalNeurons, bp.Config.Metadata.TotalLayers)
}

func setupModelTrainingSession() {
	fmt.Println("Starting to split data into training and testing sessions...")

	// Calculate split index for 80/20 split
	totalImages := len(Images)
	splitIndex := int(float64(totalImages) * 0.8)

	// Print details about the split
	fmt.Printf("Total images: %d\n", totalImages)
	fmt.Printf("80%% of images (training set): %d\n", splitIndex)
	fmt.Printf("20%% of images (testing set): %d\n", totalImages-splitIndex)

	// Loop through images and labels for training sessions (80%)
	for i := 0; i < splitIndex; i++ {
		session := createTrainingSession(i)
		TrainingSessions = append(TrainingSessions, session)
	}

	// Loop through remaining images and labels for testing sessions (20%)
	for i := splitIndex; i < totalImages; i++ {
		session := createTrainingSession(i)
		TestingSessions = append(TestingSessions, session)
	}

	fmt.Printf("Training sessions count: %d\n", len(TrainingSessions))
	fmt.Printf("Testing sessions count: %d\n", len(TestingSessions))
	fmt.Println("Completed processing all images and labels.")
}

// createTrainingSession creates a TrainingSession for a given index
func createTrainingSession(index int) blueprint.TrainingSession {
	label := Labels[index]
	expectedOutput := map[string]interface{}{
		"label": float64(label), // Convert label to float64
	}
	inputVariables := map[string]interface{}{
		"image": Images[index],
	}

	// Initialize and return the TrainingSession struct
	return blueprint.TrainingSession{
		InputVariables:   inputVariables,
		SavedLayerStates: []blueprint.LayerState{}, // Initially empty, can add states during training
		ExpectedOutput:   expectedOutput,
		Learned:          false,
	}
}

func evaluateModelPerformance() {
	fmt.Println("Evaluating model performance on the training set...")
	// Capture all six values for training set evaluation
	trainingExactAccuracy, trainingGenerousAccuracy, trainingForgivenessAccuracy,
		trainingExactErrorCount, trainingAverageGenerousError, trainingForgivenessErrorCount := bp.EvaluateModelPerformance(TrainingSessions)

	fmt.Printf("Training set exact accuracy: %.2f%%, Exact errors: %.0f\n", trainingExactAccuracy, trainingExactErrorCount)
	fmt.Printf("Training set generous accuracy: %.2f%%, Average generous error: %.2f\n", trainingGenerousAccuracy, trainingAverageGenerousError)
	fmt.Printf("Training set forgiveness accuracy: %.2f%%, Forgiveness errors: %.0f\n\n", trainingForgivenessAccuracy, trainingForgivenessErrorCount)

	fmt.Println("Evaluating model performance on the testing set...")
	// Capture all six values for testing set evaluation
	testingExactAccuracy, testingGenerousAccuracy, testingForgivenessAccuracy,
		testingExactErrorCount, testingAverageGenerousError, testingForgivenessErrorCount := bp.EvaluateModelPerformance(TestingSessions)

	fmt.Printf("Testing set exact accuracy: %.2f%%, Exact errors: %.0f\n", testingExactAccuracy, testingExactErrorCount)
	fmt.Printf("Testing set generous accuracy: %.2f%%, Average generous error: %.2f\n", testingGenerousAccuracy, testingAverageGenerousError)
	fmt.Printf("Testing set forgiveness accuracy: %.2f%%, Forgiveness errors: %.0f\n\n", testingForgivenessAccuracy, testingForgivenessErrorCount)

	// Update model metadata with accuracy and error metrics
	bp.Config.Metadata.LastTrainingAccuracy = trainingExactAccuracy
	bp.Config.Metadata.LastTestAccuracy = testingExactAccuracy
	bp.Config.Metadata.LastTestAccuracyGenerous = testingGenerousAccuracy
	bp.Config.Metadata.LastTestAccuracyForgiveness = testingForgivenessAccuracy

	// Update model metadata with error metrics
	bp.Config.Metadata.LastTrainingExactErrorCount = trainingExactErrorCount
	bp.Config.Metadata.LastTestExactErrorCount = testingExactErrorCount
	bp.Config.Metadata.LastTrainingAverageGenerousError = trainingAverageGenerousError
	bp.Config.Metadata.LastTestAverageGenerousError = testingAverageGenerousError
	bp.Config.Metadata.LastTrainingForgivenessErrorCount = trainingForgivenessErrorCount
	bp.Config.Metadata.LastTestForgivenessErrorCount = testingForgivenessErrorCount

	// Save training and testing sessions to metadata
	bp.Config.Metadata.TrainingSessions = TrainingSessions
	bp.Config.Metadata.TestingSessions = TestingSessions

	fmt.Println("Model performance evaluation completed and metadata updated.")
}
