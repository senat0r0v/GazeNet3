#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

using DlibFaceLandmarkDetector.UnityUtils;

namespace DlibFaceLandmarkDetector
{

    public class ResetDebugMode : MonoBehaviour
    {

        [InitializeOnEnterPlayMode]
        static void InitializeOnEnterPlayMode()
        {

            Debug.Log("InitializeOnEnterPlayMode()");

            Utils.setDebugMode(false);

        }
    }
}
#endif