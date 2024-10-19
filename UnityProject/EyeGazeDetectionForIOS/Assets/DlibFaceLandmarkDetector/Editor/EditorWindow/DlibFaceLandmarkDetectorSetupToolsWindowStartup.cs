using UnityEditor;
using UnityEngine;

namespace DlibFaceLandmarkDetector.Editor
{

    [InitializeOnLoad]
    public class DlibFaceLandmarkDetectorSetupToolsWindowStartup
    {

        static DlibFaceLandmarkDetectorSetupToolsWindowStartup()
        {

            EditorApplication.update -= ShowSetupToolsWindow;
            EditorApplication.update += ShowSetupToolsWindow;

            EditorApplication.playModeStateChanged -= PlayModeChanged;
            EditorApplication.playModeStateChanged += PlayModeChanged;

        }

        private static void ShowSetupToolsWindow()
        {

            //Debug.Log("DlibFaceLandmarkDetectorProjectSettings.Instance.showSetupToolsWindowFlag: " + DlibFaceLandmarkDetectorProjectSettings.Instance.showSetupToolsWindowFlag);

            var showAtStartup = DlibFaceLandmarkDetectorProjectSettings.Instance.showSetupToolsWindowFlag;

            if (showAtStartup)
            {
                DlibFaceLandmarkDetectorSetupToolsWindow.OpenSetupToolsWindow();

                DlibFaceLandmarkDetectorProjectSettings.Instance.showSetupToolsWindowFlag = false;
                EditorUtility.SetDirty(DlibFaceLandmarkDetectorProjectSettings.Instance);
            }

            EditorApplication.update -= ShowSetupToolsWindow;
        }

        private static void PlayModeChanged(PlayModeStateChange playMode)
        {
            EditorApplication.update -= ShowSetupToolsWindow;
        }
    }
}
