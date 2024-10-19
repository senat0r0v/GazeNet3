using UnityEngine;
using UnityEditor;
using System.Linq;
using System.Collections.Generic;

namespace DlibFaceLandmarkDetector.Editor
{

    public class DlibFaceLandmarkDetectorSetupToolsWindow : EditorWindow
    {

        private Vector2 _scrollPosition = Vector2.zero;


        [MenuItem("Tools/Dlib FaceLandmark Detector/Open Setup Tools", false, 1)]
        public static void OpenSetupToolsWindow()
        {

            DlibFaceLandmarkDetectorSetupToolsWindow m_Window = GetWindow<DlibFaceLandmarkDetectorSetupToolsWindow>(true, "Dlib FaceLandmark Detector | Setup Tools");
            m_Window.minSize = new Vector2(320, 790);
            //m_Window.maxSize = new Vector2(320, 800);
        }

        void OnGUI()
        {
            string resourcePath = GetResourcePath();
            //Debug.LogWarning("resourcePath " + resourcePath);
            Texture2D logo = AssetDatabase.LoadAssetAtPath<Texture2D>(resourcePath + "/EditorWindowIcon.png");
            Rect rect = GUILayoutUtility.GetRect(position.width, logo.height, GUI.skin.box);
            GUI.DrawTexture(rect, logo, ScaleMode.ScaleToFit);

            _scrollPosition = EditorGUILayout.BeginScrollView(_scrollPosition);


            EditorGUILayout.LabelField("[Setup for Example Scenes]");

            GUILayout.BeginVertical("box");
            {
                string aboutText = "Move the files from the \"DlibFaceLandmarkDetector/StreamingAssets/\" folder to the \"Assets/StreamingAssets\" folder.";
                EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                if (GUILayout.Button("Move StreamingAssets Folder"))
                {

                    MoveStreamingAssetsFolder();
                }
            }
            GUILayout.EndVertical();

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("[Setup for OpenCV Example Scenes]");

            GUILayout.BeginVertical("box");
            {
                string aboutText = "Import example scenes combined with OpenCV for Unity.";
                EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                if (GUILayout.Button("Import OpenCV Example Package"))
                {

                    DlibFaceLandmarkDetectorMenuItem.ImportDlibFaceLandmarkDetectorWithOpenCVExamplePackage();
                }
            }
            GUILayout.EndVertical();

            EditorGUILayout.Space();

            GUILayout.BeginVertical("box");
            {
                string aboutText = "Add Example Scenes to \"Scenes In Build\" in BuildSettings.";
                EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                if (GUILayout.Button("Add Example Scenes In Build"))
                {

                    AddExampleScenesInBuild();
                }
            }
            GUILayout.EndVertical();

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("[Optional]");

            GUILayout.BeginVertical("box");
            {
                string aboutText = "Set the appropriate ImportSettings for the files in the Plugins folder. Please reconfigure when you change the version of UnityEditor.";
                EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);

                if (GUILayout.Button("Set Plugin Import Settings"))
                {

                    DlibFaceLandmarkDetectorMenuItem.SetPluginImportSettings();
                }
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical("box");
            {
                string aboutText = "Import plugins with SSE4 or AVX compiler options enabled. See ReadMe.pdf for more information.";
                EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                if (GUILayout.Button("Import Extra Package"))
                {

                    DlibFaceLandmarkDetectorMenuItem.ImportExtraPackage();
                }
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical("box");
            {

                if (DlibFaceLandmarkDetectorMenuItem.ValidateUseUnsafeCode())
                {
                    string aboutText = "Add \"DLIB_DONT_USE_UNSAFE_CODE\" to ScriptCompilationDefines in BuildSettings.";
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    if (GUILayout.Button("Disable Use Unsafe Code"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseUnsafeCode(false);
                    }
                }
                else
                {                   
                    string aboutText = "Remove \"DLIB_DONT_USE_UNSAFE_CODE\" to ScriptCompilationDefines in BuildSettings.";
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    if (GUILayout.Button("Enable Use Unsafe Code"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseUnsafeCode(true);
                    }

                }
            }
            GUILayout.EndVertical();

            EditorGUILayout.Space();

            EditorGUILayout.LabelField("[WebGL Settings]");

#if UNITY_2023_3_OR_NEWER
            EditorGUI.BeginDisabledGroup(PlayerSettings.WebGL.wasm2023);
#endif
            GUILayout.BeginVertical("box");
            {
                if (DlibFaceLandmarkDetectorMenuItem.ValidateUseWebGLSIMDSupport())
                {
#if UNITY_2023_3_OR_NEWER
                    if (PlayerSettings.WebGL.wasm2023)
                    {
                        string aboutText = "PlayerSettings.WebGL.wasm2023 : " + PlayerSettings.WebGL.wasm2023;
                        EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    }
                    else
                    {
                        string aboutText = "PlayerSettings.WebGL.emscriptenArgs : " + PlayerSettings.WebGL.emscriptenArgs;
                        EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    }
#else
                    string aboutText = "PlayerSettings.WebGL.emscriptenArgs : " + PlayerSettings.WebGL.emscriptenArgs;
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
#endif
                    if (GUILayout.Button("Disable SIMD Support"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseWebGLSIMDSupport(false);
                    }
                }
                else
                {
#if UNITY_2023_3_OR_NEWER
                    if (PlayerSettings.WebGL.wasm2023)
                    {
                        string aboutText = "PlayerSettings.WebGL.wasm2023 : " + PlayerSettings.WebGL.wasm2023;
                        EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    }
                    else
                    {
                        string aboutText = "PlayerSettings.WebGL.emscriptenArgs : " + PlayerSettings.WebGL.emscriptenArgs;
                        EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    }
#else
                    string aboutText = "PlayerSettings.WebGL.emscriptenArgs : " + PlayerSettings.WebGL.emscriptenArgs;
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
#endif
                    if (GUILayout.Button("Enable SIMD Support"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseWebGLSIMDSupport(true);
                    }
                }
            }
            GUILayout.EndVertical();
#if UNITY_2023_3_OR_NEWER
            EditorGUI.EndDisabledGroup();
#endif

            GUILayout.BeginVertical("box");
            {
                if (DlibFaceLandmarkDetectorMenuItem.ValidateUseWebGLThreadsSupport())
                {
                    string aboutText = "PlayerSettings.WebGL.threadsSupport : " + PlayerSettings.WebGL.threadsSupport;
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    if (GUILayout.Button("Disable MultiThreads Support"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseWebGLThreadsSupport(false);
                    }
                }
                else
                {
                    string aboutText = "PlayerSettings.WebGL.threadsSupport : " + PlayerSettings.WebGL.threadsSupport;
                    EditorGUILayout.LabelField(aboutText, EditorStyles.textArea);
                    if (GUILayout.Button("Enable MultiThreads Support"))
                    {

                        DlibFaceLandmarkDetectorMenuItem.UseWebGLThreadsSupport(true);
                    }
                }
            }
            GUILayout.EndVertical();

            EditorGUILayout.EndScrollView();
        }

        void OnInspectorUpdate()
        {
            Repaint();
        }

        private string GetResourcePath()
        {
            MonoScript ms = MonoScript.FromScriptableObject(this);
            string path = AssetDatabase.GetAssetPath(ms);

            return path.Substring(0, path.LastIndexOf('/'));
        }

        private string GetStreamingAssetsFolderPath()
        {
            MonoScript ms = MonoScript.FromScriptableObject(this);
            string path = AssetDatabase.GetAssetPath(ms);

            return path.Substring(0, path.LastIndexOf("/Editor/EditorWindow")) + "/StreamingAssets";
        }

        private void MoveStreamingAssetsFolder()
        {
            //Only the AssetDatabase class is used for file operations, not the File.IO class.

            //Debug.Log("GetStreamingAssetsFolderPath() " + GetStreamingAssetsFolderPath());
            string DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH = GetStreamingAssetsFolderPath();
            //string DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH = "Assets/DlibFaceLandmarkDetector/StreamingAssets";
            string EDITOR_STREAMINGASSETS_PATH = "Assets/StreamingAssets";

            int errorFileCount = 0;

            string[] guids = AssetDatabase.FindAssets(null, new string[] { DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH });
            string[] paths = guids.Select(guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();
            //Debug.Log($"Search Results:\n{string.Join("\n", paths)}");

            if (paths.Length == 0)
            {
                EditorUtility.DisplayDialog("Finished", "There are no files to move.", "OK");
                return;
            }

            if (!AssetDatabase.IsValidFolder(EDITOR_STREAMINGASSETS_PATH))
            {

                string result = AssetDatabase.ValidateMoveAsset(DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH, EDITOR_STREAMINGASSETS_PATH);
                //Debug.Log("AssetDatabase.ValidateMoveAsset " + result);
                if (result == "")
                {

                    if (AssetDatabase.MoveAsset(DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH, EDITOR_STREAMINGASSETS_PATH) == "")
                    {
                        //Debug.Log("AssetDatabase.MoveAsset " + EDITOR_STREAMINGASSETS_PATH);

                        paths = null;
                    }
                    else
                    {
                        string[] splitPath = EDITOR_STREAMINGASSETS_PATH.Split('/');

                        if (AssetDatabase.CreateFolder(splitPath[0], splitPath[1]) != "")
                        {
                            //Debug.Log("succes AssetDatabase.CreateFolder " + splitPath[0] + "/" + splitPath[1]);
                        }
                    }
                }

            }

            if (paths != null)
            {
                for (int i = 0; i < paths.Length; i++)
                {
                    string path = paths[i];

                    //Debug.Log("AssetDatabase.IsValidFolder(path) " + AssetDatabase.IsValidFolder(path) + " " + path);

                    if (AssetDatabase.IsValidFolder(path))
                    {
                        string newFolderPath = EDITOR_STREAMINGASSETS_PATH + "/" + path.Substring(DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH.Length + 1);
                        //Debug.Log("newFolderPath " + newFolderPath); 

                        if (!AssetDatabase.IsValidFolder(newFolderPath))
                        {
                            string result = AssetDatabase.ValidateMoveAsset(path, newFolderPath);
                            //Debug.Log("AssetDatabase.ValidateMoveAsset " + result);
                            if (result == "")
                            {
                                if (AssetDatabase.MoveAsset(path, newFolderPath) == "")
                                {
                                    //Debug.Log("AssetDatabase.MoveAsset " + newFolderPath);
                                    for (int p = i + 1; p < paths.Length; p++)
                                    {
                                        //Debug.Log("Check SkipFile " + paths[p]);
                                        if (!paths[p].StartsWith(path)) break;
                                        i++;
                                        //Debug.Log("SkipFile " + paths[p]);
                                    }
                                    continue;
                                }
                                else
                                {
                                    string parentFolder = newFolderPath.Substring(0, newFolderPath.LastIndexOf('/'));
                                    string folderName = newFolderPath.Substring(newFolderPath.LastIndexOf('/') + 1);

                                    if (AssetDatabase.CreateFolder(parentFolder, folderName) != "")
                                    {
                                        //Debug.Log("succes AssetDatabase.CreateFolder " + parentFolder + "/" + folderName);
                                    }

                                }
                            }
                        }

                    }
                    else
                    {

                        string newPath = EDITOR_STREAMINGASSETS_PATH + "/" + path.Substring(DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH.Length + 1);
                        //Debug.Log("newPath " + newPath);

                        string result = AssetDatabase.ValidateMoveAsset(path, newPath);
                        //Debug.Log("AssetDatabase.ValidateMoveAsset " + result);
                        if (result == "")
                        {

                            if (AssetDatabase.MoveAsset(path, newPath) == "")
                            {
                                //Debug.Log("AssetDatabase.MoveAsset " + newPath);
                            }
                            else
                            {
                                errorFileCount++;
                            }
                        }
                        else
                        {
                            if (AssetDatabase.MoveAssetToTrash(newPath))
                            {
                                //Debug.Log("AssetDatabase.MoveAssetToTrash " + newPath);
                                if (AssetDatabase.MoveAsset(path, newPath) == "")
                                {
                                    //Debug.Log("AssetDatabase.MoveAsset " + newPath);
                                }
                                else
                                {
                                    errorFileCount++;
                                }
                            }
                            else
                            {
                                errorFileCount++;
                            }
                        }
                    }
                }
            }

            if (errorFileCount == 0)
            {
                if (AssetDatabase.MoveAssetToTrash(DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH))
                {
                    //Debug.Log("AssetDatabase.MoveAssetToTrash " + DLIBFACELANDMARKDETECTOR_STREAMINGASSETS_PATH);
                }

                EditorUtility.DisplayDialog("Finished", "All files were successfully moved.", "OK");

            }
            else
            {
                if (errorFileCount < 2)
                {
                    EditorUtility.DisplayDialog("Finished", "Failed to move " + errorFileCount + " file.", "OK");
                }
                else
                {
                    EditorUtility.DisplayDialog("Finished", "Failed to move " + errorFileCount + " files.", "OK");
                }
            }

            AssetDatabase.Refresh();

        }

        private string GetExamplesFolderPath()
        {
            MonoScript ms = MonoScript.FromScriptableObject(this);
            string path = AssetDatabase.GetAssetPath(ms);

            return path.Substring(0, path.LastIndexOf("/Editor/EditorWindow")) + "/Examples";
        }

        private void AddExampleScenesInBuild()
        {
            //Only the AssetDatabase class is used for file operations, not the File.IO class.

            List<string> pathList = new List<string>();

            string DLIBFACELANDMARKDETECTOR_EXAMPLES_PATH = GetExamplesFolderPath();
            //Debug.Log("GetExamplesFolderPath() " + GetExamplesFolderPath());

            if (AssetDatabase.IsValidFolder(DLIBFACELANDMARKDETECTOR_EXAMPLES_PATH))
            {

                pathList.Add(DLIBFACELANDMARKDETECTOR_EXAMPLES_PATH + "/DlibFaceLandmarkDetectorExample.unity");
                pathList.Add(DLIBFACELANDMARKDETECTOR_EXAMPLES_PATH + "/ShowSystemInfo.unity");

                foreach (var guid in AssetDatabase.FindAssets("t:Scene", new string[] { DLIBFACELANDMARKDETECTOR_EXAMPLES_PATH }))
                {
                    string path = AssetDatabase.GUIDToAssetPath(guid);
                    if (!pathList.Contains(path))
                        pathList.Add(AssetDatabase.GUIDToAssetPath(guid));
                }
            }

            string OPENCV_EXAMPLES_PATH = "Assets/DlibFaceLandmarkDetectorWithOpenCVExample";
            //Debug.Log("OPENCV_EXAMPLES_PATH " + OPENCV_EXAMPLES_PATH);

            if (AssetDatabase.IsValidFolder(OPENCV_EXAMPLES_PATH))
            {
                pathList.Add(OPENCV_EXAMPLES_PATH + "/ShowLicense.unity");

                foreach (var guid in AssetDatabase.FindAssets("t:Scene", new string[] { OPENCV_EXAMPLES_PATH }))
                {
                    string path = AssetDatabase.GUIDToAssetPath(guid);
                    if (!pathList.Contains(path))
                        pathList.Add(AssetDatabase.GUIDToAssetPath(guid));
                }
            }

            if (pathList.Count > 0)
            {

                List<EditorBuildSettingsScene> sceneList = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);

                foreach (string path in pathList)
                {
                    //Debug.Log("path " + path);
                    sceneList.Add(new EditorBuildSettingsScene(path, true));
                }

                EditorBuildSettings.scenes = sceneList.ToArray();

                EditorUtility.DisplayDialog("Finished", pathList.Count + " Example Scenes were added to \"Scenes In Build\".", "OK");
            }
            else
            {
                EditorUtility.DisplayDialog("Finished", "Example Scenes cannot be found.", "OK");
            }
        }
    }
}