// ignore: unused_import
import 'package:googleapis_auth/auth_io.dart' as auth;
import 'package:googleapis/fitness/v1.dart';

class GoogleFitService {
  final FitnessApi fitnessApi;

  GoogleFitService(this.fitnessApi);

  // Initialize Google Fit
  static Future<GoogleFitService> init() async {
    // Provide your client ID and scopes for authentication
    var clientId = ClientId('651983935961-farja7dgj61aaeu8sf2j1sqjmf1i2cbg.apps.googleusercontent.com','');
    var scopes = [FitnessApi.fitnessScope];

    // Obtain the authenticated client
    var client = await clientViaUserConsent(clientId, scopes, (url) {
      // Open the URL for the user to authenticate
      print('Please go to the following URL and grant access:');
      print('  => $url');
      print('');
    });

    // Create an instance of the Google Fit API
    var fitnessApi = FitnessApi(client);
    return GoogleFitService(fitnessApi);
  }

  // Example method to read data from Google Fit
  Future<void> readData() async {
    try {
      // Use the fitnessApi to access data
      var dataSources = await fitnessApi.users.dataSources.list("me");
      print('Data Sources: ${dataSources.dataSource}');
    } catch (e) {
      print('Error reading data: $e');
    }
  }

  // Remember to dispose the client when done
  void dispose() {
    // Perform any cleanup if necessary
    fitnessApi.client.close();
  }
}
