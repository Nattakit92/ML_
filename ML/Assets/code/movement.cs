using UnityEngine;

public class movement : MonoBehaviour
{
    public Joystick joystick;
    public Rigidbody2D car;
    public Transform car_transform;
    public float player_speed;
    public float angle;

    // Update is called once per frame
    void Update()
    {
        car.AddForce(new Vector2(Mathf.Sin(angle / 180 * Mathf.PI) * player_speed, Mathf.Cos(angle / 180 * Mathf.PI) * player_speed));
        //car.AddForce(new Vector2(joystick.Direction.x * player_speed, joystick.Direction.y * player_speed));
        float rotation = Mathf.Atan(joystick.Direction.y / joystick.Direction.x);
        car_transform.eulerAngles = new Vector3(0, 0, -angle);
        /*if (joystick.Direction.x + joystick.Direction.y != 0)
        {
            car_transform.eulerAngles = new Vector3(0, 0, (float)(rotation * 180 / 3.14) + 90);
        }*/
        if (car.velocity.sqrMagnitude > 5 * player_speed)
        {
            car.velocity *= 0.9f;
        }
    }
}
