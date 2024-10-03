import 'package:flutter/material.dart';

class LoginRegistrationPage extends StatefulWidget {
  const LoginRegistrationPage({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _LoginRegistrationPageState createState() => _LoginRegistrationPageState();
}

class _LoginRegistrationPageState extends State<LoginRegistrationPage> {
  // Controllers to get input
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  bool isLogin = true; // To toggle between login and registration
  bool isLoading = false;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  // Method to handle login
  void _login() {
    // Here you would handle the login logic (e.g., checking credentials)
    // For now, just show a success message
    setState(() {
      isLoading = true;
    });
    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Login successful')),
      );
      setState(() {
        isLoading = false;
      });
    });
  }

  // Method to handle registration
  void _register() {
    // Here you would handle the registration logic (e.g., saving user data)
    // For now, just show a success message
    setState(() {
      isLoading = true;
    });
    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Registration successful')),
      );
      setState(() {
        isLoading = false;
      });
    });
  }

  // Toggle between login and registration forms
  void toggleForm() {
    setState(() {
      isLogin = !isLogin;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(isLogin ? 'Login' : 'Register'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(labelText: 'Email'),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              obscureText: true,
              decoration: const InputDecoration(labelText: 'Password'),
            ),
            const SizedBox(height: 16),
            isLoading
                ? const CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: isLogin ? _login : _register,
                    child: Text(isLogin ? 'Login' : 'Register'),
                  ),
            const SizedBox(height: 16),
            TextButton(
              onPressed: toggleForm,
              child: Text(isLogin
                  ? 'Don\'t have an account? Register here'
                  : 'Already have an account? Login here'),
            ),
          ],
        ),
      ),
    );
  }
}
