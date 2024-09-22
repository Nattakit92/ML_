using System;
using UnityEngine;

public class Checkpoint : MonoBehaviour
{
    public Rigidbody2D car;
    public CheckpointSingle[] checkpoints;
    public float score = 0;
    public float total_score = 0;
    public int count = 0;
    public Vector2 pos_diff;
    private double last_angle = 0;
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
        pos_diff = checkpoint_pos - car_pos;
        double angle = car.transform.eulerAngles[2];

        score = -20;
        score += Vector2.Dot(car_vel, pos_diff.normalized) * 200;
        total_score += score;
        if(total_score < -500)
        {
            score -= 100;
        }

        /*score_text.text = "Score: " + Mathf.Round(score*100)/100;*/
    }
}
