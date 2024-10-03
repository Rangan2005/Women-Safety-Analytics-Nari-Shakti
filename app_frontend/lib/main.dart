import 'package:flutter/material.dart';
import 'dart:async';
import 'auth_page.dart'; // Import your authentication page

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RUDRANI',
      theme: ThemeData(
        primarySwatch: Colors.pink,
      ),
      // Define routes here
      routes: {
        '/': (context) => const WelcomePage(),
        '/auth': (context) => const LoginRegistrationPage(), // Login and registration page
      },
      initialRoute: '/', // Start at the WelcomePage
    );
  }
}

class WelcomePage extends StatelessWidget {
  const WelcomePage({super.key});

  @override
  Widget build(BuildContext context) {
    // Delay to simulate loading before navigating to AuthPage
    Timer(const Duration(seconds: 2), () {
      // Navigate to AuthPage after delay
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const LoginRegistrationPage()),
      );
    });

    return const Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'RUDRANI',
              style: TextStyle(
                fontSize: 50,
                fontWeight: FontWeight.bold,
                color: Colors.pinkAccent,
              ),
            ),
            SizedBox(height: 20),
            CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }
}
