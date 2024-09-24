using UnityEngine;
using TMPro;
using Unity.Mathematics;
using System;


public class reset : MonoBehaviour
{
    public Transform cars;
    public TextMeshProUGUI reset_text;
    private float time;
    private int reset_count = 0;
    private int checkpoint = 7;
    private int[] reset_times = new int[13] {10, 15, 20, 30, 35, 50, 60, 70, 80, 90, 100, 120, 140};
    // --------------------------------------0---1---2---3---4---5----6---7---8---9---10---11---12

    // Update is called once per frame
    void Update()
    {
        int i = 600;
        time += Time.deltaTime;
        reset_text.text = "Time : " + time.ToString();


        if (time > reset_times[checkpoint]){
            float acc_score = 0;
            float hscore = Single.MinValue;
            int hcheckpoint = 0;
            int hcount = 0;
            checkpoint = 0;
            reset_count++;
            foreach (Transform cars_10 in cars)
            {
                foreach (Transform car in cars_10)
                {
                    if (checkpoint > hcheckpoint){
                        hcount = 1;
                        hcheckpoint = checkpoint;
                    } else if (checkpoint == hcheckpoint){
                        hcount++;
                    }
                    if (car.GetComponent<Checkpoint>().total_score > hscore)
                    {
                        hscore = car.GetComponent<Checkpoint>().total_score;
                        checkpoint = car.GetComponent<Checkpoint>().count;
                    }

                    acc_score += car.GetComponent<Checkpoint>().total_score;
                    car.GetComponent<movement>().car.position = new Vector3(-7, -1, 0);
                    car.GetComponent<movement>().car.velocity = new Vector3(0, 0, 0);
                    car.GetComponent<movement>().angle = 0;
                    car.GetComponent<Checkpoint>().count = 0;
                    car.GetComponent<Checkpoint>().score = 0;
                    car.GetComponent<Checkpoint>().total_score = 0;
                    cars.GetComponent<Cars_score>().step[800] = 1;
                    i++;
                }
            }
            checkpoint = math.min(checkpoint, 12);
            Debug.Log("Gen " + reset_count.ToString() + " score : " + (acc_score/(i - 600)).ToString() + ", checkpoint : " + hcheckpoint.ToString() + ", count : " + hcount.ToString() + ", next time : " + reset_times[checkpoint].ToString());
            time = 0;
        }
    }
}
