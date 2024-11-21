using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class VideoReceiver : MonoBehaviour
{
    public RawImage rawImage;
    private TcpClient client;
    private NetworkStream stream;
    private byte[] buffer = new byte[8192];
    private StringBuilder messageBuilder = new StringBuilder();

    void Start()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5000);
            stream = client.GetStream();
            Debug.Log("Connected to server.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to connect: {e.Message}");
        }
    }

    void Update()
    {
        if (stream != null && stream.DataAvailable)
        {
            try
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                Debug.Log($"Bytes read: {bytesRead}");
                string messagePart = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                messageBuilder.Append(messagePart);

                string[] messages = messageBuilder.ToString().Split('\n');
                for (int i = 0; i < messages.Length - 1; i++)
                {
                    byte[] bytes = Convert.FromBase64String(messages[i]);
                    Texture2D texture = new Texture2D(1, 1);
                    texture.LoadImage(bytes);
                    rawImage.texture = texture;
                    Debug.Log("Frame updated.");
                }
                messageBuilder.Clear();
                messageBuilder.Append(messages[messages.Length - 1]);
            }
            catch (Exception e)
            {
                Debug.LogError($"Error processing data: {e.Message}");
            }
        }
    }

    void OnDestroy()
    {
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
    }
}
