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
    public float reward = 0;
    private int count = 0;
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
            score += 5;
            if (count > 11)
            {
                count = 0;
                Debug.Log("Done");
            }
        }
    }
    void Update()
    {
        reward = checkpoints[count].transform.position.sqrMagnitude - car.position.sqrMagnitude;
        if (reward < 0) 
        {
            reward = -reward;
        }
        reward /= 500;
        reward = 0.03f - reward;
        score += reward;
        /*score_text.text = "Score: " + Mathf.Round(score*100)/100;*/
    }
}
