import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart'; // For Google Fit API authentication
// Implement your Google Fit service logic here
import 'package:googleapis/fitness/v1.dart';
import 'package:http/http.dart' as http;

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  _DashboardPageState createState() => _DashboardPageState();
}

class GoogleFitService {
  static final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: [
      'https://www.googleapis.com/auth/fitness.activity.read',
      'https://www.googleapis.com/auth/fitness.heart_rate.read',
    ],
  );

  // Sign in to Google and fetch health data
  static Future<Map<String, String>> getHealthData() async {
    try {
      final account = await _googleSignIn.signIn();
      final authHeaders = await account?.authHeaders;
      if (authHeaders == null) throw Exception('Failed to sign in to Google');

      final client = GoogleHttpClient(authHeaders);
      final fitnessApi = FitnessApi(client);

      // Fetch heart rate data from Google Fit API
      var datasets = await fitnessApi.users.dataset.aggregate('me' as AggregateRequest, AggregateRequest(
        aggregateBy: [AggregateBy(dataTypeName: 'com.google.heart_rate.bpm')],
        bucketByTime: BucketByTime(durationMillis: 86400000), // 1 day
        startTimeMillis: DateTime.now().tostring().subtract(const Duration(days: 1)).millisecondsSinceEpoch,
        endTimeMillis: DateTime.now().millisecondsSinceEpoch,
      ) as String);

      // Extract heart rate and pulse from datasets
      String heartRate = datasets.bucket[0].dataset[0].point.isNotEmpty
          ? datasets.bucket[0].dataset[0].point[0].value[0].fpVal.toString()
          : 'Unavailable';

      String pulseRate = datasets.bucket[0].dataset[0].point.isNotEmpty
          ? datasets.bucket[0].dataset[0].point[0].value[0].fpVal.toString()
          : 'Unavailable';

      return {'heartRate': heartRate, 'pulseRate': pulseRate};
    } catch (e) {
      print('Error fetching Google Fit data: $e');
      return {'heartRate': 'Error', 'pulseRate': 'Error'};
    }
  }
}

// Custom HTTP Client for Google Fit API requests
class GoogleHttpClient extends http.BaseClient {
  final Map<String, String> _headers;
  final http.Client _client = http.Client();

  GoogleHttpClient(this._headers);

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      _client.send(request..headers.addAll(_headers));
}

class _DashboardPageState extends State<DashboardPage> {
  // List of recipients for SOS
  List<String> recipients = ["Friend 1", "Family Member", "Emergency Contact"];
  String selectedRecipient = "";

  // Example health data
  String heartRate = 'Unknown';
  String pulseRate = 'Unknown';

  // Google Fit API Data Fetch
  Future<void> fetchHealthData() async {
    try {
      final healthData = await GoogleFitService.getHealthData();
      setState(() {
        heartRate = healthData['heartRate']!;
        pulseRate = healthData['pulseRate']!;
      });
    } catch (e) {
      print('Error fetching health data: $e');
    }
  }

  @override
  void initState() {
    super.initState();
    // Fetch initial Google Fit data
    fetchHealthData();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Rudrani Dashboard'),  // Changed name to "Rudrani"
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // SOS Button
            Center(
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                ),
                icon: const Icon(Icons.warning, color: Colors.white),
                label: const Text("Send SOS", style: TextStyle(color: Colors.white)),
                onPressed: () {
                  _sendSOS();
                },
              ),
            ),
            const SizedBox(height: 20),

            // List of Recipients for SOS
            const Text(
              "Send SOS to:",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            DropdownButton<String>(
              hint: const Text("Select Recipient"),
              value: selectedRecipient.isEmpty ? null : selectedRecipient,
              onChanged: (newValue) {
                setState(() {
                  selectedRecipient = newValue!;
                });
              },
              items: recipients.map((recipient) {
                return DropdownMenuItem<String>(
                  value: recipient,
                  child: Text(recipient),
                );
              }).toList(),
            ),

            // Heart Rate and Pulse Tracker
            const SizedBox(height: 30),
            const Text(
              "Health Tracker:",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            Card(
              margin: const EdgeInsets.only(top: 10),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    ListTile(
                      title: Text('Heart Rate: $heartRate bpm'),
                      trailing: const Icon(Icons.favorite, color: Colors.red),
                    ),
                    ListTile(
                      title: Text('Pulse Rate: $pulseRate bpm'),
                      trailing: const Icon(Icons.healing, color: Colors.blue),
                    ),
                    ElevatedButton(
                      onPressed: fetchHealthData,
                      child: const Text('Refresh Health Data'),
                    )
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Example SOS sending method
  void _sendSOS() {
    if (selectedRecipient.isNotEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('SOS sent to $selectedRecipient')),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a recipient')),
      );
    }
  }
}
