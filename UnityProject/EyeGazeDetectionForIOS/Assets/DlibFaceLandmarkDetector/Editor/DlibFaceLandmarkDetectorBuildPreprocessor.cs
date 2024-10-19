using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace DlibFaceLandmarkDetector.Editor
{

    public class DlibFaceLandmarkDetectorBuildPreprocessor : IPreprocessBuildWithReport
    {

        public void OnPreprocessBuild(BuildReport report)
        {
            string[] guids = UnityEditor.AssetDatabase.FindAssets("DlibFaceLandmarkDetectorBuildPreprocessor");
            if (guids.Length == 0)
            {
                Debug.LogWarning("SetPluginImportSettings Faild : DlibFaceLandmarkDetectorBuildPreprocessor.cs is missing.");
                return;
            }
            string dlibFaceLandmarkDetectorFolderPath = AssetDatabase.GUIDToAssetPath(guids[0]).Substring(0, AssetDatabase.GUIDToAssetPath(guids[0]).LastIndexOf("/Editor/DlibFaceLandmarkDetectorBuildPreprocessor.cs"));

            string pluginsFolderPath = dlibFaceLandmarkDetectorFolderPath + "/Plugins";
            //Debug.Log("pluginsFolderPath " + pluginsFolderPath);

            string extraFolderPath = dlibFaceLandmarkDetectorFolderPath + "/Extra";
            //Debug.Log("extraFolderPath " + extraFolderPath);

            Debug.Log("DlibFaceLandmarkDetectorBuildPreprocessor " + report.summary.platform);

            switch (report.summary.platform)
            {
#if UNITY_2017_3_OR_NEWER
                case BuildTarget.StandaloneOSX:
#else
                case BuildTarget.StandaloneOSXUniversal:
                case BuildTarget.StandaloneOSXIntel:
                case BuildTarget.StandaloneOSXIntel64:
#endif
                    DlibFaceLandmarkDetectorMenuItem.SetOSXPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.StandaloneWindows:
                case BuildTarget.StandaloneWindows64:

                    DlibFaceLandmarkDetectorMenuItem.SetWindowsPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.iOS:

                    DlibFaceLandmarkDetectorMenuItem.SetIOSPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.Android:

                    DlibFaceLandmarkDetectorMenuItem.SetAndroidPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
#if UNITY_2019_2_OR_NEWER
                case BuildTarget.StandaloneLinux64:
#else
                case BuildTarget.StandaloneLinux:
                case BuildTarget.StandaloneLinux64:
                case BuildTarget.StandaloneLinuxUniversal:
#endif

                    DlibFaceLandmarkDetectorMenuItem.SetLinuxPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.WebGL:

                    DlibFaceLandmarkDetectorMenuItem.SetWebGLPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.WSAPlayer:

                    DlibFaceLandmarkDetectorMenuItem.SetUWPPluginImportSettings(pluginsFolderPath, extraFolderPath);
                    break;
                case BuildTarget.NoTarget:

                    break;
                default:

                    break;
            }

        }

        public int callbackOrder { get { return 0; } }
    }
}

