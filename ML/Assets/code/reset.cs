using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class reset : MonoBehaviour
{
    public Transform cars;
    private float time;

    // Update is called once per frame
    void Update()
    {
        int i = 600;
        time += Time.deltaTime;
        if (time > 5) 
        {
            foreach (Transform cars_10 in cars)
            {
                foreach (Transform car in cars_10)
                {
                    if (car.GetComponent<Checkpoint>().total_score < -50) 
                    {
                        car.GetComponent<movement>().car.position = new Vector3(-7, -1, 0);
                        car.GetComponent<movement>().angle = 0;
                        car.GetComponent<Checkpoint>().count = 0;
                        car.GetComponent<Checkpoint>().score = 0;
                        car.GetComponent<Checkpoint>().total_score = 0;
                        cars.GetComponent<Cars_score>().step[i] = 0;
                        i++;
                    }
                }
            }   
        }
    }
}
