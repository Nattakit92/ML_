using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class reset : MonoBehaviour
{
    public Transform cars;
    private float time = 0;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        time += Time.deltaTime;
        if (time > 6) 
        {
            time = 0;
            foreach (Transform cars_10 in cars)
            {
                foreach (Transform car in cars_10)
                {
                    car.GetComponent<movement>().angle = 0;
                    car.GetComponent<movement>().car.position = new Vector3(-7,-1,0);
                    car.GetComponent<Checkpoint>().score = 0;
                }
            }
            cars.GetComponent<Cars_score>().scores = new float[100];
        }
    }
}
