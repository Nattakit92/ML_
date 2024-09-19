using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckpointSingle : MonoBehaviour
{
    public Vector2 location;
    public Checkpoint checkpoints;
    public bool collide = false;

    private void OnTriggerEnter2D(Collider2D collision)
    {
        collision.transform.GetComponent<Checkpoint>().CheckpointCheck(this);
    }
    public void SetCheckpoint(Checkpoint checkpoints)
    {
        this.checkpoints = checkpoints;
    }
}
