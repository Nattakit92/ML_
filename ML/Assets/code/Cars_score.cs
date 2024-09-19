using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

public class Cars_score : MonoBehaviour
{
    public Transform cars;
    public float[] scores = new float[100];
    public float[] rotation = new float[100];
    private int count = 0;
    private void Update()
    {
        foreach (Transform cars_10 in cars) 
        {
            foreach(Transform car in cars_10)
            {
                car.GetComponent<movement>().angle = rotation[count];
                scores[count] = car.GetComponent<Checkpoint>().score;
                count++;
            }
        }
        count = 0;
    }
}
