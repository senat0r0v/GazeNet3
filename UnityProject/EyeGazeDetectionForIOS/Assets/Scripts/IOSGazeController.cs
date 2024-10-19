using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using DlibFaceLandmarkDetector;
using Unity.Barracuda;
using System.IO;
using System.Linq;
using UnityEngine.UI;
#if UNITY_IOS
using UnityEngine.Networking;
#endif

public class IOSGazeController : MonoBehaviour
{
    // -------------------------------
    // ======= Grid Settings =========
    // -------------------------------

    public GameObject cubePrefab;
    public Vector3 prefabScale = new Vector3(1f, 1f, 1f); // Scaling for prefabs

    private GameObject regionsParent;
    private Dictionary<string, GameObject> gridCubes = new Dictionary<string, GameObject>();
    private Dictionary<string, Color> regionColors = new Dictionary<string, Color>();

    // -------------------------------
    // ===== Eye Extraction Settings ==
    // -------------------------------

    public string shapePredictorPath;
    private FaceLandmarkDetector landmarkDetector;
    private bool cameraReady = false; // Renamed from isInitialized

    [Header("Padding Settings")]
    [Tooltip("Percentage of the eye height to add as padding on the top and bottom.")]
    [Range(0f, 0.5f)]
    public float verticalPaddingFactor = 0.5f;

    [Tooltip("Percentage of the eye width to add as padding on the sides.")]
    [Range(0f, 0.5f)]
    public float horizontalPaddingFactor = 0.35f;

    [Header("Capture Settings")]
    [Tooltip("Scale factor to control how much of the eye is captured.")]
    [Range(0.5f, 2.0f)]
    public float eyeCaptureScale = 1.0f; // New tunable variable

    // -------------------------------
    // ===== ONNX Model Settings =====
    // -------------------------------

    public NNModel onnxModelAsset;
    private Model runtimeModel;
    private IWorker worker;

    // -------------------------------
    // ===== UI Settings =============
    // -------------------------------

    [Header("UI Settings")]
    [Tooltip("RawImage UI element to display the extracted eye image.")]
    public RawImage eyeDisplay; // Added for displaying the eye texture

    [Tooltip("Text UI element to display logs, warnings, and messages.")]
    public Text legacyLogText; // New field for legacy text logs

    // -------------------------------
    // ===== Additional Fields ========
    // -------------------------------

    private Texture2D displayTexture; // Texture assigned to RawImage to persist across frames

    // Coroutine handles to manage different coroutines
    private Coroutine landmarkLoadingCoroutine;
    private Coroutine gazeProcessingCoroutine;

    // -------------------------------
    // ====== Class Definitions =======
    // -------------------------------

    private readonly string[] classes = new string[]
    {
        "Center",
        "Down",
        "Up"
    };

    // -------------------------------
    // ====== New Tunable Variables ===
    // -------------------------------

    [Header("Processing Settings")]
    [Tooltip("Interval in seconds between each gaze processing step.")]
    [Range(0.05f, 1.0f)]
    public float processingInterval = 0.1f; // New tunable variable

    [Tooltip("Number of recent predictions to average for determining the most frequent gaze direction.")]
    [Range(1, 100)]
    public int averageWindowSize = 10; // New tunable variable

    [Header("Rotation Settings")]
    [Tooltip("Degrees to rotate the camera image counter-clockwise on iOS (must be 0, 90, 180, or 270).")]
    [Range(0, 270)]
    public int rotationDegrees = 90; // New tunable variable

    // Buffer to store recent predictions
    private Queue<string> predictionBuffer = new Queue<string>();

    // -------------------------------
    // ====== Camera Handling ========
    // -------------------------------

    private WebCamTexture webCamTexture;

    // -------------------------------
    // ====== Unity Methods ==========
    // -------------------------------

    /// <summary>
    /// Unity's Start method as a coroutine to handle asynchronous initialization.
    /// </summary>
    /// <returns></returns>
    IEnumerator Start()
    {
        // Initialize grid and region colors
        InitializeRegionColors();
        GenerateGrid();

        // Initialize landmark detector
        landmarkLoadingCoroutine = StartCoroutine(InitializeLandmarkDetector());
        yield return landmarkLoadingCoroutine;

        // Proceed only if landmark detector is initialized
        if (landmarkDetector != null)
        {
            // Initialize webcam with improved handling
            StartCamera();

            // Load ONNX model
            InitializeONNXModel();

            // Initialize displayTexture if eyeDisplay is assigned
            if (eyeDisplay != null)
            {
                displayTexture = new Texture2D(224, 224, TextureFormat.RGB24, false);
                eyeDisplay.texture = displayTexture;
            }
            else
            {
                Debug.LogWarning("RawImage for eye display is not assigned in the Inspector.");
            }

            // Initialize legacy log text if assigned
            if (legacyLogText != null)
            {
                legacyLogText.text = ""; // Clear any existing text
                Application.logMessageReceived += HandleLog; // Subscribe to log events
            }
            else
            {
                Debug.LogWarning("Legacy Log Text is not assigned in the Inspector.");
            }
        }
        else
        {
            Debug.LogError("Landmark detector initialization failed. Gaze processing will not proceed.");
        }
    }

    void OnDestroy()
    {
        // Dispose resources
        if (landmarkDetector != null)
            landmarkDetector.Dispose();

        if (webCamTexture != null && webCamTexture.isPlaying)
            webCamTexture.Stop();

        if (worker != null)
            worker.Dispose();

        if (displayTexture != null)
            Destroy(displayTexture);

        // Stop the gaze processing coroutine if it's running
        if (gazeProcessingCoroutine != null)
            StopCoroutine(gazeProcessingCoroutine);

        // Unsubscribe from log events
        Application.logMessageReceived -= HandleLog;
    }

    #region Grid Initialization

    void InitializeRegionColors()
    {
        // Assign default colors to each class
        foreach (string region in classes)
        {
            regionColors.Add(region, Color.white);
        }
    }

    void GenerateGrid()
    {
        regionsParent = new GameObject("Regions");
        for (int i = 0; i < classes.Length; i++)
        {
            string region = classes[i];
            float zPosition = (i - 1) * 2f; // Positions at z=-2, 0, 2
            GameObject cube = Instantiate(cubePrefab, new Vector3(0f, 0f, zPosition), Quaternion.identity);
            cube.transform.localScale = prefabScale;
            cube.transform.parent = regionsParent.transform;

            Renderer cubeRenderer = cube.GetComponent<Renderer>();
            cubeRenderer.material.color = regionColors[region];

            gridCubes[region] = cube;
        }
    }

    #endregion

    #region Landmark Detector Initialization

    /// <summary>
    /// Coroutine to initialize the landmark detector.
    /// Ensures that the detector is fully initialized before proceeding.
    /// </summary>
    /// <returns></returns>
    IEnumerator InitializeLandmarkDetector()
    {
#if UNITY_IOS
        // On iOS, StreamingAssets are packed inside the app bundle and require UnityWebRequest to access
        yield return StartCoroutine(LoadShapePredictorIOS());
#else
        // For other platforms, load directly from StreamingAssets
        LoadShapePredictor();
        yield return null;
#endif
    }

#if UNITY_IOS
    /// <summary>
    /// Coroutine to load the shape predictor on iOS using UnityWebRequest.
    /// </summary>
    /// <returns></returns>
    IEnumerator LoadShapePredictorIOS()
    {
        // Construct the relative path to the shape predictor
        string relativePath = "DlibFaceLandmarkDetector/sp_human_face_68.dat";
        string combinedPath = Path.Combine(Application.streamingAssetsPath, relativePath);

        // Prepend "file://" to the path for iOS
        string fullPath = "file://" + combinedPath;

        Debug.Log($"Loading shape predictor from: {fullPath}");

        // Use UnityWebRequest to read the binary data
        UnityWebRequest www = UnityWebRequest.Get(fullPath);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Failed to load shape predictor on iOS: " + www.error);
            yield break;
        }

        // Save the downloaded data to a temporary file
        string tempPath = Path.Combine(Application.temporaryCachePath, "sp_human_face_68.dat");
        try
        {
            File.WriteAllBytes(tempPath, www.downloadHandler.data);
            shapePredictorPath = tempPath;
            Debug.Log("Shape predictor saved to temporary path: " + shapePredictorPath);
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to write shape predictor to temporary path: " + ex.Message);
            yield break;
        }

        // Initialize the landmark detector
        try
        {
            landmarkDetector = new FaceLandmarkDetector(shapePredictorPath);
            Debug.Log("Shape predictor loaded successfully on iOS.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to initialize landmark detector on iOS: " + ex.Message);
        }
    }
#endif

    /// <summary>
    /// Loads the shape predictor for non-iOS platforms.
    /// </summary>
    void LoadShapePredictor()
    {
        // Construct the full path to the shape predictor
        shapePredictorPath = Path.Combine(Application.streamingAssetsPath, "DlibFaceLandmarkDetector", "sp_human_face_68.dat");

        Debug.Log($"Loading shape predictor from: {shapePredictorPath}");

        if (!File.Exists(shapePredictorPath))
        {
            Debug.LogError("Shape predictor file does not exist at path: " + shapePredictorPath);
            return;
        }

        // Load the Dlib shape predictor
        try
        {
            landmarkDetector = new FaceLandmarkDetector(shapePredictorPath);
            Debug.Log("Shape predictor loaded successfully.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to load shape predictor: " + ex.Message);
        }
    }

    #endregion

    #region Improved Webcam Initialization

    /// <summary>
    /// Initializes the camera with improved handling for iOS.
    /// </summary>
    void StartCamera()
    {
#if UNITY_IOS
        StartCoroutine(RequestCameraPermission());
#else
        StartCoroutine(InitializeCameraRoutine());
#endif
    }

#if UNITY_IOS
    /// <summary>
    /// Coroutine to request camera permissions on iOS.
    /// </summary>
    /// <returns></returns>
    IEnumerator RequestCameraPermission()
    {
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            yield return StartCoroutine(InitializeCameraIOS());
        }
        else
        {
            Debug.LogError("Camera permission not granted.");
        }
    }

    /// <summary>
    /// Coroutine to initialize the camera on iOS.
    /// </summary>
    /// <returns></returns>
    IEnumerator InitializeCameraIOS()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("No webcam detected on this device.");
            yield break;
        }

        WebCamDevice[] devices = WebCamTexture.devices;
        string frontCamName = "";

        for (int i = 0; i < devices.Length; i++)
        {
            if (devices[i].isFrontFacing)
            {
                frontCamName = devices[i].name;
                break;
            }
        }

        if (string.IsNullOrEmpty(frontCamName))
        {
            Debug.LogWarning("Front camera not found. Using the first available camera.");
            frontCamName = devices[0].name;
        }

        webCamTexture = new WebCamTexture(frontCamName);

        // On iOS, use lower resolution to optimize performance
        webCamTexture.requestedWidth = 640;
        webCamTexture.requestedHeight = 480;

        webCamTexture.Play();
        yield return StartCoroutine(WaitForCameraToInitialize());
    }
#endif

    /// <summary>
    /// Coroutine to initialize the camera on non-iOS platforms.
    /// </summary>
    /// <returns></returns>
    IEnumerator InitializeCameraRoutine()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("No webcam detected on this device.");
            yield break;
        }

        WebCamDevice[] devices = WebCamTexture.devices;
        string frontCamName = "";

        for (int i = 0; i < devices.Length; i++)
        {
            if (devices[i].isFrontFacing)
            {
                frontCamName = devices[i].name;
                break;
            }
        }

        if (string.IsNullOrEmpty(frontCamName))
        {
            Debug.LogWarning("Front camera not found. Using the first available camera.");
            frontCamName = devices[0].name;
        }

        webCamTexture = new WebCamTexture(frontCamName);

        // For other platforms, you can use higher resolution if needed
        webCamTexture.requestedWidth = 1280;
        webCamTexture.requestedHeight = 720;

        webCamTexture.Play();
        yield return StartCoroutine(WaitForCameraToInitialize());
    }

    /// <summary>
    /// Coroutine that waits for the webcam to initialize and then starts gaze processing.
    /// </summary>
    /// <returns></returns>
    IEnumerator WaitForCameraToInitialize()
    {
        float timeout = Time.time + 10f; // Increased timeout for better reliability
        while (!webCamTexture.didUpdateThisFrame)
        {
            if (Time.time > timeout)
            {
                Debug.LogError("Camera initialization timed out.");
                webCamTexture.Stop();
                webCamTexture = null;
                yield break;
            }
            yield return null;
        }

        Debug.Log("Webcam initialized. Resolution: " + webCamTexture.width + "x" + webCamTexture.height);
        cameraReady = true;

        // Start the gaze processing coroutine
        gazeProcessingCoroutine = StartCoroutine(ProcessGazeCoroutine());
    }

    #endregion

    #region ONNX Model Initialization

    /// <summary>
    /// Initializes the ONNX model using Unity's Barracuda library.
    /// </summary>
    void InitializeONNXModel()
    {
        if (onnxModelAsset == null)
        {
            Debug.LogError("ONNX model asset is not assigned.");
            return;
        }

        try
        {
            runtimeModel = ModelLoader.Load(onnxModelAsset);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);
            Debug.Log("ONNX model loaded successfully.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to load ONNX model: " + ex.Message);
        }
    }

    #endregion

    #region Update Loop

    // Removed the gaze processing from Update()
    void Update()
    {
        // If there are other frame-based updates, they can be handled here.
    }

    #endregion

    #region Gaze Processing Coroutine

    /// <summary>
    /// Coroutine that processes gaze direction at regular intervals.
    /// </summary>
    /// <returns></returns>
    IEnumerator ProcessGazeCoroutine()
    {
        while (cameraReady)
        {
            // Capture the current frame from the webcam
            Texture2D currentFrame = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGB24, false);
            currentFrame.SetPixels(webCamTexture.GetPixels());
            currentFrame.Apply();

#if UNITY_IOS
            // Rotate the image by rotationDegrees counter-clockwise to correct iOS automatic rotation
            Texture2D rotatedFrame = RotateTextureByDegrees(currentFrame, rotationDegrees);
            if (rotatedFrame != null)
            {
                Destroy(currentFrame); // Clean up the original frame
                currentFrame = rotatedFrame;
                Debug.Log($"Rotated camera frame by {rotationDegrees} degrees counter-clockwise for iOS.");
            }
            else
            {
                Debug.LogError("Failed to rotate camera frame.");
            }
#endif

            // Detect and process landmarks
            DetectAndProcessGaze(currentFrame);

            // Clean up
            Destroy(currentFrame);

            // Wait for the specified processing interval before the next processing
            yield return new WaitForSeconds(processingInterval);
        }
    }

    #endregion

    #region Gaze Detection and Processing

    /// <summary>
    /// Detects gaze direction based on the provided image.
    /// </summary>
    /// <param name="image">The image captured from the webcam.</param>
    void DetectAndProcessGaze(Texture2D image)
    {
        if (landmarkDetector == null)
        {
            Debug.LogError("Landmark detector is not initialized.");
            return;
        }

        // Detect facial bounding boxes
        landmarkDetector.SetImage(image);
        List<Rect> faceRects = landmarkDetector.Detect(); // Detect faces

        if (faceRects != null && faceRects.Count > 0)
        {
            // For this implementation, we'll process only the first detected face
            Rect faceRect = faceRects[0];
            Debug.Log($"Processing face at: {faceRect}");

            // Detect facial landmarks for the detected face
            List<Vector2> landmarks = landmarkDetector.DetectLandmark(faceRect); // Pass face bounding box

            if (landmarks != null && landmarks.Count == 68) // Ensure all 68 landmarks are detected
            {
                // Adjust landmarks for Unity's coordinate system
                for (int i = 0; i < landmarks.Count; i++)
                {
                    landmarks[i] = FlipYCoordinate(landmarks[i], image.height);
                }

                // Extract the right eye image
                Texture2D eyeTexture = ExtractRightEye(image, landmarks);

                if (eyeTexture != null)
                {
                    // Display the eye image on the RawImage UI
                    if (eyeDisplay != null)
                    {
                        // Copy the eyeTexture data to displayTexture to persist it
                        DisplayTexture(eyeTexture);
                    }
                    else
                    {
                        Debug.LogWarning("RawImage for eye display is not assigned.");
                    }

                    // Prepare the eye image for the model
                    Tensor inputTensor = PrepareInputTensor(eyeTexture);

                    if (inputTensor != null)
                    {
                        // Predict gaze direction
                        string predictedRegion = PredictGazeDirection(inputTensor);

                        // Add prediction to the buffer
                        AddPrediction(predictedRegion);

                        // If enough predictions are collected, compute the mode and update the grid
                        if (predictionBuffer.Count >= averageWindowSize)
                        {
                            string averagedPrediction = ComputeMode(predictionBuffer);
                            UpdateGridBasedOnPrediction(averagedPrediction);

                            // Log the averaged prediction
                            Debug.Log($"Averaged Predicted Gaze Direction: {averagedPrediction}");

                            // Clear the buffer for the next set of predictions
                            predictionBuffer.Clear();
                        }

                        // Clean up
                        inputTensor.Dispose();
                    }

                    Destroy(eyeTexture);
                }
            }
            else
            {
                Debug.LogError("No face landmarks detected or incorrect number of landmarks.");
            }
        }
        else
        {
            Debug.LogError("No faces detected.");
        }
    }

    /// <summary>
    /// Adds a new prediction to the buffer.
    /// </summary>
    /// <param name="prediction">The predicted gaze direction.</param>
    void AddPrediction(string prediction)
    {
        if (predictionBuffer.Count >= averageWindowSize)
        {
            predictionBuffer.Dequeue();
        }
        predictionBuffer.Enqueue(prediction);
    }

    /// <summary>
    /// Computes the mode (most frequent element) of the predictions in the buffer.
    /// </summary>
    /// <param name="buffer">Queue containing recent predictions.</param>
    /// <returns>The most frequent prediction.</returns>
    string ComputeMode(Queue<string> buffer)
    {
        if (buffer.Count == 0)
            return "Unknown";

        // Count the frequency of each prediction
        Dictionary<string, int> frequencyDict = new Dictionary<string, int>();
        foreach (string prediction in buffer)
        {
            if (frequencyDict.ContainsKey(prediction))
                frequencyDict[prediction]++;
            else
                frequencyDict[prediction] = 1;
        }

        // Find the prediction with the highest frequency
        int maxCount = 0;
        string mode = "Unknown";
        foreach (var kvp in frequencyDict)
        {
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                mode = kvp.Key;
            }
        }

        return mode;
    }

    /// <summary>
    /// Flips the Y-coordinate to match Unity's coordinate system.
    /// </summary>
    /// <param name="point">The original point.</param>
    /// <param name="imageHeight">Height of the image.</param>
    /// <returns>Flipped point.</returns>
    Vector2 FlipYCoordinate(Vector2 point, int imageHeight)
    {
        return new Vector2(point.x, imageHeight - point.y);
    }

    /// <summary>
    /// Extracts and processes the right eye region from the image.
    /// </summary>
    /// <param name="image">The full image.</param>
    /// <param name="landmarks">Detected facial landmarks.</param>
    /// <returns>Processed eye texture.</returns>
    Texture2D ExtractRightEye(Texture2D image, List<Vector2> landmarks)
    {
        // Dlib's 68 landmarks: Right eye (42-47, 0-based indexing)
        // Indices: 42,43,44,45,46,47
        List<Vector2> rightEyeLandmarks = new List<Vector2>();
        for (int i = 42; i <= 47; i++)
        {
            rightEyeLandmarks.Add(landmarks[i]);
        }

        // Determine the bounding rectangle for the right eye
        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;

        foreach (Vector2 point in rightEyeLandmarks)
        {
            if (point.x < minX) minX = point.x;
            if (point.y < minY) minY = point.y;
            if (point.x > maxX) maxX = point.x;
            if (point.y > maxY) maxY = point.y;
        }

        // Calculate the width and height of the bounding box
        float eyeWidth = maxX - minX;
        float eyeHeight = maxY - minY;

        // Apply the eyeCaptureScale to control the amount captured
        float scaledEyeWidth = eyeWidth * eyeCaptureScale;
        float scaledEyeHeight = eyeHeight * eyeCaptureScale;

        // Calculate padding based on the scaled dimensions
        float paddingY = scaledEyeHeight * verticalPaddingFactor;
        float paddingX = scaledEyeWidth * horizontalPaddingFactor;

        // Adjust the bounding box with padding
        minX = Mathf.Max(0, minX - paddingX);
        minY = Mathf.Max(0, minY - paddingY);
        maxX = Mathf.Min(image.width, maxX + paddingX);
        maxY = Mathf.Min(image.height, maxY + paddingY);

        // Define the rectangle to crop
        Rect eyeRect = new Rect(minX, minY, maxX - minX, maxY - minY);

        // Debugging: Log the cropping rectangle
        Debug.Log($"Cropping eye region: {eyeRect}");

        // Crop the right eye region
        Texture2D croppedEye = CropTexture(image, eyeRect);

        if (croppedEye == null)
        {
            Debug.LogError("Failed to crop the right eye region.");
            return null;
        }

        // Resize the cropped eye to desired size (224x224)
        Texture2D resizedEye = ResizeTexture(croppedEye, 224, 224);
        Destroy(croppedEye); // Clean up the cropped texture

        if (resizedEye == null)
        {
            Debug.LogError("Failed to resize the cropped eye image.");
            return null;
        }

        return resizedEye;
    }

    /// <summary>
    /// Crops the texture based on the provided rectangle.
    /// </summary>
    /// <param name="source">Source texture.</param>
    /// <param name="rect">Rectangle to crop.</param>
    /// <returns>Cropped texture.</returns>
    Texture2D CropTexture(Texture2D source, Rect rect)
    {
        try
        {
            // Ensure the rectangle is within the source texture bounds
            rect.x = Mathf.Clamp(rect.x, 0, source.width);
            rect.y = Mathf.Clamp(rect.y, 0, source.height);
            rect.width = Mathf.Clamp(rect.width, 1, source.width - rect.x);
            rect.height = Mathf.Clamp(rect.height, 1, source.height - rect.y);

            Texture2D cropped = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
            Color[] pixels = source.GetPixels((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
            cropped.SetPixels(pixels);
            cropped.Apply();
            return cropped;
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Error cropping texture: " + ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Resizes the texture to the specified dimensions.
    /// </summary>
    /// <param name="source">Source texture.</param>
    /// <param name="newWidth">New width.</param>
    /// <param name="newHeight">New height.</param>
    /// <returns>Resized texture.</returns>
    Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        try
        {
            RenderTexture rt = new RenderTexture(newWidth, newHeight, 24);
            RenderTexture currentRT = RenderTexture.active;
            RenderTexture.active = rt;
            Graphics.Blit(source, rt);
            Texture2D resized = new Texture2D(newWidth, newHeight, TextureFormat.RGB24, false);
            resized.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
            resized.Apply();
            RenderTexture.active = currentRT;
            rt.Release();
            return resized;
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Error resizing texture: " + ex.Message);
            return null;
        }
    }

    #endregion

    #region Tensor Preparation and Prediction

    /// <summary>
    /// Prepares the input tensor for the ONNX model.
    /// </summary>
    /// <param name="eyeTexture">The processed eye texture.</param>
    /// <returns>Tensor ready for model input.</returns>
    Tensor PrepareInputTensor(Texture2D eyeTexture)
    {
        // Define ImageNet normalization constants
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        // Get pixels from the image
        Color[] pixels = eyeTexture.GetPixels();

        // Initialize a float array for the tensor data in NHWC order
        float[] floatValues = new float[3 * eyeTexture.width * eyeTexture.height];

        for (int i = 0; i < pixels.Length; i++)
        {
            // Normalize each channel and arrange in NHWC
            // Remove the division by 255f since Color.r/g/b are already 0-1
            float r = pixels[i].r;
            float g = pixels[i].g;
            float b = pixels[i].b;

            // Apply ImageNet normalization
            floatValues[i * 3 + 0] = (r - mean[0]) / std[0]; // Red
            floatValues[i * 3 + 1] = (g - mean[1]) / std[1]; // Green
            floatValues[i * 3 + 2] = (b - mean[2]) / std[2]; // Blue
        }

        // Create the tensor in NHWC format (Batch, Height, Width, Channels)
        Tensor input = new Tensor(1, eyeTexture.height, eyeTexture.width, 3, floatValues);
        Debug.Log($"Input Tensor Shape: {input.shape}");

        return input;
    }

    /// <summary>
    /// Predicts the gaze direction based on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Predicted gaze direction as a string.</returns>
    string PredictGazeDirection(Tensor input)
    {
        if (worker == null)
        {
            Debug.LogError("Barracuda worker is not initialized.");
            return "Unknown";
        }

        try
        {
            // Execute the model
            worker.Execute(input);
            Tensor output = worker.PeekOutput();

            // Get the index with the highest probability
            float maxVal = output[0];
            int maxIndex = 0;
            for (int i = 1; i < output.length; i++)
            {
                if (output[i] > maxVal)
                {
                    maxVal = output[i];
                    maxIndex = i;
                }
            }

            // Map the index to the corresponding class
            string predictedRegion = (maxIndex >= 0 && maxIndex < classes.Length) ? classes[maxIndex] : "Unknown";

            Debug.Log($"Predicted Gaze Direction Index: {maxIndex}, Region: {predictedRegion}");

            output.Dispose();

            return predictedRegion;
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Error during model prediction: " + ex.Message);
            return "Unknown";
        }
    }

    #endregion

    #region Grid Update Based on Prediction

    /// <summary>
    /// Updates the grid based on the predicted gaze direction.
    /// </summary>
    /// <param name="predictedRegion">The predicted gaze direction region.</param>
    void UpdateGridBasedOnPrediction(string predictedRegion)
    {
        // First, reset all cubes to their original color
        foreach (var kvp in gridCubes)
        {
            Renderer cubeRenderer = kvp.Value.GetComponent<Renderer>();
            cubeRenderer.material.color = regionColors[kvp.Key];
        }

        // Then, update the cube in the predicted region to red
        if (gridCubes.ContainsKey(predictedRegion))
        {
            Renderer cubeRenderer = gridCubes[predictedRegion].GetComponent<Renderer>();
            cubeRenderer.material.color = Color.red;
        }
        else
        {
            Debug.LogWarning("Predicted region not found in grid: " + predictedRegion);
        }
    }

    #endregion

    #region Texture Display

    /// <summary>
    /// Copies the eyeTexture data to the displayTexture for persistence.
    /// </summary>
    /// <param name="eyeTexture">The processed eye texture.</param>
    void DisplayTexture(Texture2D eyeTexture)
    {
        if (displayTexture == null)
        {
            Debug.LogError("Display texture is not initialized.");
            return;
        }

        // Ensure the sizes match
        if (displayTexture.width != eyeTexture.width || displayTexture.height != eyeTexture.height)
        {
            Debug.LogWarning("Display texture size does not match eye texture size. Resizing display texture.");
            Destroy(displayTexture);
            displayTexture = new Texture2D(eyeTexture.width, eyeTexture.height, TextureFormat.RGB24, false);
            eyeDisplay.texture = displayTexture;
        }

        // Copy pixels from eyeTexture to displayTexture
        displayTexture.SetPixels(eyeTexture.GetPixels());
        displayTexture.Apply();
    }

    #endregion

    #region Log Handling

    /// <summary>
    /// Handles incoming log messages and displays them in the legacy text UI.
    /// </summary>
    /// <param name="logString">The log message.</param>
    /// <param name="stackTrace">The stack trace associated with the log.</param>
    /// <param name="type">The type of log message.</param>
    void HandleLog(string logString, string stackTrace, LogType type)
    {
        if (legacyLogText == null)
            return;

        // Format the log message
        string formattedLog = $"<color={GetColorHex(type)}>{type.ToString()}: {logString}</color>\n";

        // Append the log to the legacy text
        legacyLogText.text += formattedLog;

        // Optional: Implement a maximum number of lines to prevent the text from growing indefinitely
        int maxLines = 50;
        string[] lines = legacyLogText.text.Split('\n');
        if (lines.Length > maxLines)
        {
            legacyLogText.text = string.Join("\n", lines.Skip(lines.Length - maxLines));
        }
    }

    /// <summary>
    /// Converts LogType to corresponding hex color string.
    /// </summary>
    /// <param name="type">The type of log message.</param>
    /// <returns>Hex color string.</returns>
    string GetColorHex(LogType type)
    {
        switch (type)
        {
            case LogType.Error:
            case LogType.Exception:
                return "#FF0000"; // Red
            case LogType.Warning:
                return "#FFFF00"; // Yellow
            case LogType.Log:
                return "#FFFFFF"; // White
            case LogType.Assert:
                return "#FF00FF"; // Magenta
            default:
                return "#FFFFFF"; // Default to White
        }
    }

    #endregion

    #region Helper Methods

#if UNITY_IOS
    /// <summary>
    /// Rotates a Texture2D by specified degrees counter-clockwise.
    /// Supports only multiples of 90 degrees.
    /// </summary>
    /// <param name="source">The source texture.</param>
    /// <param name="degrees">Degrees to rotate counter-clockwise (must be 0, 90, 180, or 270).</param>
    /// <returns>The rotated texture.</returns>
    Texture2D RotateTextureByDegrees(Texture2D source, int degrees)
    {
        try
        {
            switch (degrees)
            {
                case 0:
                    return source;
                case 90:
                    return RotateTexture90CounterClockwise(source);
                case 180:
                    return RotateTexture180(source);
                case 270:
                    return RotateTexture270CounterClockwise(source);
                default:
                    Debug.LogError("Unsupported rotation degrees. Must be 0, 90, 180, or 270.");
                    return null;
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Error rotating texture: " + ex.Message);
            return null;
        }
    }

    /// <summary>
    /// Rotates a Texture2D by 90 degrees counter-clockwise.
    /// </summary>
    Texture2D RotateTexture90CounterClockwise(Texture2D source)
    {
        int width = source.width;
        int height = source.height;

        Texture2D rotated = new Texture2D(height, width, source.format, false);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                rotated.SetPixel(y, width - x - 1, source.GetPixel(x, y));
            }
        }

        rotated.Apply();
        return rotated;
    }

    /// <summary>
    /// Rotates a Texture2D by 180 degrees.
    /// </summary>
    Texture2D RotateTexture180(Texture2D source)
    {
        int width = source.width;
        int height = source.height;

        Texture2D rotated = new Texture2D(width, height, source.format, false);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                rotated.SetPixel(width - x - 1, height - y - 1, source.GetPixel(x, y));
            }
        }

        rotated.Apply();
        return rotated;
    }

    /// <summary>
    /// Rotates a Texture2D by 270 degrees counter-clockwise.
    /// Equivalent to 90 degrees clockwise.
    /// </summary>
    Texture2D RotateTexture270CounterClockwise(Texture2D source)
    {
        int width = source.width;
        int height = source.height;

        Texture2D rotated = new Texture2D(height, width, source.format, false);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                rotated.SetPixel(height - y - 1, x, source.GetPixel(x, y));
            }
        }

        rotated.Apply();
        return rotated;
    }
#endif

    #endregion
}
