using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

public class Cars_score : MonoBehaviour
{
    public Transform cars;
    public int[] step = new int[700];
    public float[] rotation = new float[100];
    private int count = 0;
    private void Update()
    {
        foreach (Transform cars_10 in cars) 
        {
            foreach(Transform car in cars_10)
            {
                car.GetComponent<movement>().angle = rotation[count];
                step[count] = (int)car.GetComponent<Checkpoint>().score;
                step[count + 100] = (int)car.GetComponent<movement>().car.velocity.x;
                step[count + 200] = (int)car.GetComponent<movement>().car.velocity.y;
                step[count + 300] = (int)car.GetComponent<Checkpoint>().pos_diff.x;
                step[count + 400] = (int)car.GetComponent<Checkpoint>().pos_diff.y;
                step[count + 500] = (int)car.GetComponent<Checkpoint>().count;
                count++;
            }
        }
        count = 0;
    }
}
