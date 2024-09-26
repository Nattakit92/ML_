using System;
using UnityEngine;
using UnityEngine.Rendering;

public class Cars_score : MonoBehaviour
{
    public Transform cars;
    public int[] step = new int[801];
    public float[] rotation = new float[100];
    private int i = 0;

    private void Start(){
        step = new int[801];
    }
    private void Update()
    {
        foreach (Transform cars_10 in cars) 
        {
            foreach(Transform car in cars_10)
            {
                car.GetComponent<movement>().angle = rotation[i];
                step[i] = (int)car.GetComponent<Checkpoint>().score;
                step[i + 100] = (int)Math.Round(car.GetComponent<movement>().car.velocity.x);
                step[i + 200] = (int)Math.Round(car.GetComponent<movement>().car.velocity.y);
                step[i + 300] = (int)Math.Round(car.GetComponent<movement>().car.position.x);
                step[i + 400] = (int)Math.Round(car.GetComponent<movement>().car.position.y);
                step[i + 500] = (int)Math.Round(car.GetComponent<Checkpoint>().checkpoint_pos.x);
                step[i + 600] = (int)Math.Round(car.GetComponent<Checkpoint>().checkpoint_pos.y);
                step[i + 700] = (int)car.GetComponent<Checkpoint>().count;
                i++;
            }
        }
        i = 0;
    }
}
