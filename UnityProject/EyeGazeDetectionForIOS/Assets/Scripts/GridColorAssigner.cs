using UnityEngine;
using System.Collections;
using UnityEngine.UI; // For Texture2D
using System.IO; // For Directory and File operations
using System; // For Random and String operations
using NativeGalleryNamespace; // Ensure you are using NativeGallery plugin

public class GridColorAssigner : MonoBehaviour
{
    public GameObject cubePrefab;
    public int gridWidth = 20;  // Width of the grid (x-axis)
    public int gridHeight = 40; // Height of the grid (z-axis)
    public Vector3 prefabScale = new Vector3(1f, 1f, 1f); // Scaling for prefabs
    public float cubeSpawnHeight = 1.5f; // Height to spawn the cube above the grid

    private GameObject regionsParent;
    private GameObject currentCube;
    private WebCamTexture webcamTexture;
    private bool cameraReady = false;

    void Start()
    {
        regionsParent = new GameObject("Regions");
        GenerateGrid();
        SpawnCube();

        StartCamera();
    }

    void OnDestroy()
    {
        if (webcamTexture != null && webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
        }
    }

    void StartCamera()
    {
#if UNITY_IOS
        StartCoroutine(RequestCameraPermission());
#else
        InitializeCamera();
#endif
    }

    IEnumerator RequestCameraPermission()
    {
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            InitializeCamera();
        }
        else
        {
            Debug.LogError("Camera permission not granted.");
        }
    }

    void InitializeCamera()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogWarning("No camera detected");
            return;
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
            Debug.LogWarning("Front camera not found");
            return;
        }

        webcamTexture = new WebCamTexture(frontCamName);
        webcamTexture.Play();
        StartCoroutine(WaitForCameraToInitialize());
    }

    IEnumerator WaitForCameraToInitialize()
    {
        float timeout = Time.time + 5f;
        while (!webcamTexture.didUpdateThisFrame)
        {
            if (Time.time > timeout)
            {
                Debug.LogWarning("Camera initialization timed out");
                webcamTexture.Stop();
                webcamTexture = null;
                yield break;
            }
            yield return null;
        }
        cameraReady = true;
        Debug.Log("Camera is ready");
    }

    public void OnCubeTapped(Vector3 cubePosition)
    {
        string region = GetRegion((int)cubePosition.x, (int)cubePosition.z);
        Debug.Log($"White cube at X: {cubePosition.x}, Z: {cubePosition.z} was tapped in {region} region.");

        Destroy(currentCube);

        StartCoroutine(TakePhoto(region));

        Invoke("SpawnCube", 0.25f);
    }

    IEnumerator TakePhoto(string region)
    {
        if (!cameraReady)
        {
            Debug.LogWarning("Camera is not ready");
            yield break;
        }

        yield return new WaitForEndOfFrame();

        Texture2D photo = new Texture2D(webcamTexture.width, webcamTexture.height);
        photo.SetPixels(webcamTexture.GetPixels());
        photo.Apply();

        byte[] bytes = photo.EncodeToJPG();
        string filename = $"{GenerateRandomString(8)}_{region}.jpg";

        // Save the image to the iOS photo gallery
        NativeGallery.SaveImageToGallery(bytes, "MyAppPhotos", filename);
        Debug.Log($"Photo saved to gallery with filename: {filename}");

        Destroy(photo);
        yield return null;
    }

    void GenerateGrid()
    {
        int halfWidth = gridWidth / 2;
        int halfHeight = gridHeight / 2;
        float widthToHeightRatio = (float)gridWidth / gridHeight;

        for (int x = -halfWidth; x <= halfWidth; x++)
        {
            for (int z = -halfHeight; z <= halfHeight; z++)
            {
                GameObject cube = Instantiate(cubePrefab, new Vector3(x, 0, z), Quaternion.identity);
                cube.transform.localScale = prefabScale;
                cube.transform.parent = regionsParent.transform;

                Renderer cubeRenderer = cube.GetComponent<Renderer>();

                float verticalStretch = 5f * 2.5f * 1.58f * widthToHeightRatio;
                float horizontalStretch = 10f * 0.58f;

                // Determine if cube is inside the oval (Center)
                bool insideOval = (Mathf.Pow(x, 2) / Mathf.Pow(horizontalStretch, 2)) +
                                  (Mathf.Pow(z, 2) / Mathf.Pow(verticalStretch, 2)) <= 1;

                // Region logic and coloring
                if (insideOval)
                {
                    cubeRenderer.material.color = new Color(0.65f, 0.16f, 0.16f); // Center (Brown)
                }
                else if (z > 0 && Mathf.Abs(x) < z * widthToHeightRatio)
                {
                    cubeRenderer.material.color = Color.yellow; // Up (Yellow)
                }
                else if (z < 0 && Mathf.Abs(x) < -z * widthToHeightRatio)
                {
                    cubeRenderer.material.color = Color.green;  // Down (Green)
                }
                else if (x < 0 && Mathf.Abs(z) < -x / widthToHeightRatio)
                {
                    cubeRenderer.material.color = Color.blue;  // Left (Blue)
                }
                else if (x > 0 && Mathf.Abs(z) < x / widthToHeightRatio)
                {
                    cubeRenderer.material.color = Color.red;  // Right (Red)
                }
                else
                {
                    // Fix the unknown regions, assign them to nearest neighboring regions
                    if (x < 0 && z > 0)
                    {
                        cubeRenderer.material.color = Color.yellow; // Part of Up
                    }
                    else if (x > 0 && z > 0)
                    {
                        cubeRenderer.material.color = Color.yellow; // Part of Up
                    }
                    else if (x < 0 && z < 0)
                    {
                        cubeRenderer.material.color = Color.green;  // Part of Down
                    }
                    else if (x > 0 && z < 0)
                    {
                        cubeRenderer.material.color = Color.green;  // Part of Down
                    }
                }
            }
        }
    }

    void SpawnCube()
    {
        int x = UnityEngine.Random.Range(-gridWidth / 2, gridWidth / 2);
        int z = UnityEngine.Random.Range(-gridHeight / 2, gridHeight / 2);

        currentCube = Instantiate(cubePrefab, new Vector3(x, cubeSpawnHeight, z), Quaternion.identity);
        currentCube.transform.localScale = prefabScale;
        currentCube.AddComponent<BoxCollider>();
        currentCube.AddComponent<CubeTapHandler>();
        currentCube.transform.parent = regionsParent.transform;

        Renderer cubeRenderer = currentCube.GetComponent<Renderer>();
        cubeRenderer.material.color = Color.white;

        string region = GetRegion(x, z);
        Debug.Log($"White cube spawned at X: {x}, Z: {z} in {region} region.");
    }

    string GetRegion(int x, int z)
    {
        float widthToHeightRatio = (float)gridWidth / gridHeight;
        float verticalStretch = 5f * 2.5f * 1.58f * widthToHeightRatio;
        float horizontalStretch = 10f * 0.58f;

        bool insideOval = (Mathf.Pow(x, 2) / Mathf.Pow(horizontalStretch, 2)) +
                          (Mathf.Pow(z, 2) / Mathf.Pow(verticalStretch, 2)) <= 1;

        if (insideOval)
        {
            return "Center";
        }
        else if (z > 0 && Mathf.Abs(x) < z * widthToHeightRatio)
        {
            return "Up";
        }
        else if (z < 0 && Mathf.Abs(x) < -z * widthToHeightRatio)
        {
            return "Down";
        }
        else if (x < 0 && Mathf.Abs(z) < -x / widthToHeightRatio)
        {
            return "Left";
        }
        else if (x > 0 && Mathf.Abs(z) < x / widthToHeightRatio)
        {
            return "Right";
        }

        // Assign any ambiguous region to the closest direction
        if (x < 0 && z > 0)
        {
            return "Up";
        }
        else if (x > 0 && z > 0)
        {
            return "Up";
        }
        else if (x < 0 && z < 0)
        {
            return "Down";
        }
        else if (x > 0 && z < 0)
        {
            return "Down";
        }

        return "Unknown";
    }

    string GenerateRandomString(int length)
    {
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        char[] stringChars = new char[length];
        System.Random random = new System.Random();

        for (int i = 0; i < length; i++)
        {
            stringChars[i] = chars[random.Next(chars.Length)];
        }

        return new string(stringChars);
    }
}

public class CubeTapHandler : MonoBehaviour
{
    private GridColorAssigner gridColorAssigner;

    void Start()
    {
        gridColorAssigner = FindObjectOfType<GridColorAssigner>();
    }

    void OnMouseDown()
    {
        gridColorAssigner.OnCubeTapped(transform.position);
    }
}