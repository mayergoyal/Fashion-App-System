import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { ScrollView } from "react-native-gesture-handler";

export default function FashionRec() {
  const [image, setImage] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  // Pick Image
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.IMAGE,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri); // Fix for expo-image-picker v13+
    }
  };

  // Send Image to Flask API
  const getRecommendations = async () => {
    if (!image) {
      Alert.alert("Error", "Please select an image first.");
      return;
    }

    let formData = new FormData();
    formData.append("image", {
      uri: image,
      name: "image.jpg",
      type: "image/jpeg",
    });

    try {
      let response = await fetch("http://192.168.109.188:5000/upload", {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (!response.ok) throw new Error("Failed to fetch recommendations");

      let data = await response.json();
      setRecommendations(data.matches);
    } catch (error) {
      console.error("Error:", error);
      Alert.alert("Error", "Failed to get recommendations.");
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={pickImage} style={styles.button}>
        <Text style={styles.buttonText}>Pick an Image</Text>
      </TouchableOpacity>

      {image && <Image source={{ uri: image }} style={styles.image} />}

      <TouchableOpacity onPress={getRecommendations} style={styles.button}>
        <Text style={styles.buttonText}>Get Recommendations</Text>
      </TouchableOpacity>

      {recommendations.length > 0 && (
        <ScrollView contentContainerStyle={styles.recommendationsContainer}>
          {recommendations.map((item, index) => (
            <Image
              key={index}
              source={{ uri: item }}
              style={styles.recommededimg}
            />
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f9fa",
  },
  button: {
    padding: 12,
    backgroundColor: "#007bff",
    margin: 10,
    borderRadius: 5,
  },
  buttonText: { color: "#fff", fontWeight: "bold" },
  image: { width: 200, height: 200, margin: 10, borderRadius: 10 },

  recommededimg: {
    width: 100,
    height: 100,
    margin: 5,
    borderRadius: 10,
  },
  recommendationsContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
  },
});
