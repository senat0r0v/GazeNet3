using System.IO;
using UnityEditor;
using UnityEngine;

namespace DlibFaceLandmarkDetector.Editor
{

    public class DlibFaceLandmarkDetectorProjectSettings : ScriptableObject
    {
        private static DlibFaceLandmarkDetectorProjectSettings _instance;

        public static DlibFaceLandmarkDetectorProjectSettings Instance
        {
            get
            {
                if (_instance == null)
                {

                    string path = GetEditorFolderPath() + "/DlibFaceLandmarkDetectorProjectSettings.asset";
                    _instance = AssetDatabase.LoadAssetAtPath<DlibFaceLandmarkDetectorProjectSettings>(path);

                    if (_instance == null)
                    {
                        _instance = ScriptableObject.CreateInstance<DlibFaceLandmarkDetectorProjectSettings>();
                        _instance.showSetupToolsWindowFlag = true;
                        AssetDatabase.CreateAsset(_instance, path);
                    }
                }

                return _instance;
            }
        }

        public bool showSetupToolsWindowFlag;

        static string GetEditorFolderPath()
        {

            string[] guids = UnityEditor.AssetDatabase.FindAssets("DlibFaceLandmarkDetectorProjectSettings");
            if (guids.Length == 0)
            {
                Debug.LogWarning("DlibFaceLandmarkDetectorProjectSettings.cs is missing.");
                return null;
            }
            string editorFolderPath = AssetDatabase.GUIDToAssetPath(guids[0]).Substring(0, AssetDatabase.GUIDToAssetPath(guids[0]).LastIndexOf("/DlibFaceLandmarkDetectorProjectSettings"));

            //Debug.Log("editorFolderPath " + editorFolderPath);

            return editorFolderPath;
        }
    }
}
