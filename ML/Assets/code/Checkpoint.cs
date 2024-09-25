using System;
using UnityEngine;

public class Checkpoint : MonoBehaviour
{
    public Rigidbody2D car;
    public CheckpointSingle[] checkpoints;
    public float score = 0;
    public float total_score = 0;
    public int count = 0;
    public Vector2 checkpoint_pos;
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
            score += 1000 * count;
            if (count > 11)
            {
                score += 10000;
                count = 0;
                Debug.Log("Done");
            }
        }
    }
    void Update()
    {
        checkpoint_pos = checkpoints[count].transform.position;
        Vector2 car_pos = car.position;
        Vector2 car_vel = car.velocity;
        Vector2 pos_diff = checkpoint_pos - car_pos;

        score = -100;
        score += count * 100;
        score += Vector2.Dot(car_vel, pos_diff.normalized) * 200;
        if ((Math.Pow(car_pos[0], 2) + (4 * Math.Pow(car_pos[1], 2))) > 64)
        {
            score -= 200;
        }

        total_score += score;
        // if(total_score < -500)
        // {
        //     score -= 100;
        // }

        // score_text.text = "Score: " + Mathf.Round(score*100)/100;
    }
}
