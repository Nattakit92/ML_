using System.Collections;
using System.Collections.Generic;
using System.Xml.Serialization;
using TMPro;
using UnityEngine;
using UnityEngine.SocialPlatforms.Impl;

public class Checkpoint : MonoBehaviour
{
    public Rigidbody2D car;
    public CheckpointSingle[] checkpoints;
    public TextMeshProUGUI score_text;
    public float score = 0;
    public float total_score = 0;
    public int count = 0;
    void Start()
    {
        score_text.text = "Score: 1";
        foreach(CheckpointSingle checkpoint in checkpoints)
        {
            checkpoint.SetCheckpoint(this);
        }
    }
    public void CheckpointCheck(CheckpointSingle checkpoint)
    {
        if (count.ToString() == checkpoint.transform.name){
            count++;
            score += 100;
            if (count > 11)
            {
                count = 0;
                Debug.Log("Done");
            }
            total_score = count*5;
        }
    }
    void Update()
    {
        Vector2 checkpoint_pos = checkpoints[count].transform.position;
        Vector2 car_pos = car.position;
        Vector2 car_vel = car.velocity;
        Vector2 pos_diff = checkpoint_pos - car_pos;

        score = Vector2.Dot(car_vel.normalized, pos_diff.normalized) * 10;
        total_score += score;
        if(total_score < -50)
        {
            score = -1000;
        }

        /*score_text.text = "Score: " + Mathf.Round(score*100)/100;*/
    }
}
