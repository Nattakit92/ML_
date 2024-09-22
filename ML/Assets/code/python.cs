using System;
using System.Net.Sockets;
using UnityEngine;

public class python : MonoBehaviour
{
    // IP and port of the Python server
    public Cars_score cars_score;
    public string host = "127.0.0.1"; // Localhost
    public int port = 65432;          // Must match the Python server port

    // Socket object
    private TcpClient client;

    void Start()
    {
        ConnectToPythonServer();
    }

    void Update()
    {
        SendDataToPython(cars_score.step);
        for(int i = 600;i < 700; i++)
        {
            cars_score.step[i] = 0;
        }
    }

    void ConnectToPythonServer()
    {
        try
        {
            client = new TcpClient(host, port);
            Debug.Log("Connected to Python server.");
        }
        catch (Exception e)
        {
            Debug.LogError("Error connecting to server: " + e.Message);
        }
    }

    void SendDataToPython(int[] messageArray)
    {
        if (client == null) return;

        try
        {
            // Convert the float array to byte array
            byte[] data = new byte[messageArray.Length * sizeof(int)];
            Buffer.BlockCopy(messageArray, 0, data, 0, data.Length);

            // Send data to the server
            NetworkStream stream = client.GetStream();
            stream.Write(data, 0, data.Length);

            // Receive response from the server
            byte[] responseData = new byte[100 * sizeof(float)];
            int bytes = stream.Read(responseData, 0, responseData.Length);

            // Convert the response byte array back to float array
            float[] responseArray = new float[100];
            Buffer.BlockCopy(responseData, 0, responseArray, 0, responseData.Length);

            // Assign the response array to movement
            cars_score.rotation = responseArray;
        }
        catch (Exception e)
        {
            Debug.LogError("Error sending data: " + e.Message);
        }
    }

    private void OnApplicationQuit()
    {
        if (client != null)
        {
            client.Close();
        }
    }
}
