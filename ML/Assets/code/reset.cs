using UnityEngine;
using TMPro;


public class reset : MonoBehaviour
{
    public Transform cars;
    public TextMeshProUGUI reset_text;
    private float time;
    private int reset_count = 0;

    // Update is called once per frame
    void Update()
    {
        int i = 600;
        time += Time.deltaTime;

        if (time > 40){
            float acc_score = 0;
            reset_count++;
            reset_text.text = "Reset Count : " + reset_count.ToString();
            foreach (Transform cars_10 in cars)
            {
                foreach (Transform car in cars_10)
                {
                    acc_score += car.GetComponent<Checkpoint>().total_score;
                    car.GetComponent<movement>().car.position = new Vector3(-7, -1, 0);
                    car.GetComponent<movement>().car.velocity = new Vector3(0, 0, 0);
                    car.GetComponent<movement>().angle = 0;
                    car.GetComponent<Checkpoint>().count = 0;
                    car.GetComponent<Checkpoint>().score = 0;
                    car.GetComponent<Checkpoint>().total_score = 0;
                    cars.GetComponent<Cars_score>().step[i] = 1;
                    i++;
                }
            }
            Debug.Log("Gen : " + reset_count.ToString() + " : " + (acc_score/time).ToString());
            time = 0;
        }
    }
}
