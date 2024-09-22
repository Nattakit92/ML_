using System;
using System.Collections;
using System.Collections.Generic;
using System.Xml.Serialization;
using UnityEngine;
using UnityEngine.SocialPlatforms.Impl;

public class Checkpoint : MonoBehaviour
{
    public Rigidbody2D car;
    public CheckpointSingle[] checkpoints;
    public float score = 0;
    public float total_score = 0;
    public int count = 0;
    public Vector2 pos_diff;
    void Start()
    {
        foreach(CheckpointSingle checkpoint in checkpoints)
        {
            checkpoint.SetCheckpoint(this);
        }
    }
    public void CheckpointCheck(CheckpointSingle checkpoint)
    {
        if (count.ToString() == checkpoint.transform.name){
            count++;
            score += 200 * count;
            if (count > 11)
            {
                score += 1000;
                count = 0;
                Debug.Log("Done");
            }
            total_score = count * 5;
        }
    }
    void Update()
    {
        Vector2 checkpoint_pos = checkpoints[count].transform.position;
        Vector2 car_pos = car.position;
        Vector2 car_vel = car.velocity;
        Vector2 car_angle = new Vector2((float)Math.Sin(car.GetComponent<movement>().angle),(float)Math.Cos(car.GetComponent<movement>().angle));
        pos_diff = checkpoint_pos - car_pos;

        score = Vector2.Dot(car_angle.normalized, pos_diff.normalized) * 100;
        total_score += score;
        total_score -= 1;
        if(total_score < -100)
        {
            score = -1000;
        }

        /*score_text.text = "Score: " + Mathf.Round(score*100)/100;*/
    }
}
