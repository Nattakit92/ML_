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
                step[i + 100] = (int)car.GetComponent<movement>().car.velocity.x * 100;
                step[i + 200] = (int)car.GetComponent<movement>().car.velocity.y * 100;
                step[i + 300] = (int)car.GetComponent<movement>().car.position.x * 100;
                step[i + 400] = (int)car.GetComponent<movement>().car.position.y * 100;
                step[i + 500] = (int)car.GetComponent<Checkpoint>().checkpoint_pos.x * 100;
                step[i + 600] = (int)car.GetComponent<Checkpoint>().checkpoint_pos.y * 100;
                step[i + 700] = (int)car.GetComponent<Checkpoint>().count;
                i++;
            }
        }
        i = 0;
    }
}
